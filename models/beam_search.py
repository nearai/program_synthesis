import sys
import os
import time
import collections

import torch
from torch.autograd import Variable
import torch.nn.functional as F


# TODO: Make an argument (dataset arg?).
MAX_DECODER_LENGTH = 100


def expand_by_beam(v, beam_size):
    # v: batch size x ...
    # output: (batch size * beam size) x ...
    return (v.unsqueeze(1).repeat(1, beam_size, *([1] * (v.dim() - 1))).
            view(-1, *v.shape[1:]))


class BeamSearchMemory(object):
    '''Batched memory used in beam search.'''
    __slots__ = ()

    def expand_by_beam(self, beam_size):
        '''Return a copy of self where each item has been replicated
        `beam_search` times.'''
        raise NotImplementedError


class BeamSearchState(object):
    '''Batched recurrent state used in beam search.'''
    __slots__ = ()

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch_size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        raise NotImplementedError


BeamSearchResult = collections.namedtuple('BeamSearchResult', ['sequence', 'total_log_prob', 'log_probs'])


def beam_search(batch_size,
                enc,
                masked_memory,
                decode_fn,
                beam_size,
                cuda=False,
                max_decoder_length=MAX_DECODER_LENGTH,
                return_attention=False,
                return_beam_search_result=False,
                volatile=True):
    # enc: batch size x hidden size
    # memory: batch size x sequence length x hidden size
    tt = torch.cuda if cuda else torch
    prev_tokens = Variable(tt.LongTensor(
        batch_size).fill_(0), volatile=volatile)
    prev_probs = Variable(tt.FloatTensor(
        batch_size, 1).fill_(0), volatile=volatile)
    prev_hidden = enc
    finished = [[] for _ in range(batch_size)]
    result = [[BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0)
               for _ in range(beam_size)] for _ in range(batch_size)]
    batch_finished = [False for _ in range(batch_size)]
    # b_idx: 0, ..., 0, 1, ..., 1, ..., b, ..., b
    # where b is the batch size, and each group of numbers has as many elements
    # as the beam size.
    b_idx = Variable(
        torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, beam_size).view(-1), volatile=volatile)

    prev_masked_memory = masked_memory.expand_by_beam(beam_size)

    attn_list = [] if return_attention else None
    for step in range(max_decoder_length):
        hidden, logits = decode_fn(prev_tokens, prev_hidden,
                                prev_masked_memory if step > 0 else
                                masked_memory, attentions=attn_list)

        logit_size = logits.size(1)
        # log_probs: batch size x beam size x vocab size
        log_probs = F.log_softmax(logits, dim=-1).view(batch_size, -1, logit_size)
        total_log_probs = log_probs + prev_probs.unsqueeze(2)
        # log_probs_flat: batch size x beam_size * vocab_size
        log_probs_flat = total_log_probs.view(batch_size, -1)
        # indices: batch size x beam size
        # Each entry is in [0, beam_size * vocab_size)
        prev_probs, indices = log_probs_flat.topk(min(beam_size, log_probs_flat.size(1)), dim=1)
        # prev_tokens: batch_size * beam size
        # Each entry indicates which token should be added to each beam.
        prev_tokens = (indices % logit_size).view(-1)
        # This takes a lot of time... about 50% of the whole thing.
        indices = indices.cpu()
        # k_idx: batch size x beam size
        # Each entry is in [0, beam_size), indicating which beam to extend.
        k_idx = (indices / logit_size)
        idx = torch.stack([b_idx, k_idx.view(-1)])
        # prev_hidden: (batch size * beam size) x hidden size
        # Contains the hidden states which produced the top-k (k = beam size)
        # tokens, and should be extended in the step.
        prev_hidden = hidden.select_for_beams(batch_size, idx)

        prev_result = result
        result = [[] for _ in range(batch_size)]
        can_stop = True
        k_idx = k_idx.data.numpy()
        indices = indices.data.numpy()
        for batch_id in range(batch_size):
            if batch_finished[batch_id]:
                continue
            # print(step, finished[batch_id])
            if len(finished[batch_id]) >= beam_size:
                # If last in finished has bigger log prob then best in topk, stop.
                if finished[batch_id][-1].total_log_prob > prev_probs.data[batch_id][0]:
                    batch_finished[batch_id] = True
                    continue
            for idx in range(beam_size):
                token = indices[batch_id, idx] % logit_size
                kidx = k_idx[batch_id, idx]
                # print(step, batch_id, idx, 'token', token, kidx, 'prev', prev_result[batch_id][kidx], prev_probs.data[batch_id][idx])
                if token == 1:  # 1 == </S>
                    finished[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence,
                        total_log_prob=prev_probs.data[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs.data[batch_id, kidx, token]]))
                    result[batch_id].append(BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0))
                    prev_probs.data[batch_id][idx] = float('-inf')
                else:
                    result[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence + [token],
                        total_log_prob=prev_probs.data[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs.data[batch_id, kidx, token]]))
                    can_stop = False
            if len(finished[batch_id]) >= beam_size:
                # Sort and clip.
                finished[batch_id] = sorted(
                    finished[batch_id], key=lambda x: -x.total_log_prob)[:beam_size]
        if can_stop:
            break

    for batch_id in range(batch_size):
        # If there is deficit in finished, fill it in with highest probable results.
        if len(finished[batch_id]) < beam_size:
            i = 0
            while i < beam_size and len(finished[batch_id]) < beam_size:
                if result[batch_id][i]:
                    finished[batch_id].append(result[batch_id][i])
                i += 1

    if not return_beam_search_result:
        for batch_id in range(batch_size):
            finished[batch_id] = [x.sequence for x in finished[batch_id]]

    if return_attention:
        # all elements of attn_list: (batch size * beam size) x input length
        attn_list[0] = expand_by_beam(attn_list[0], beam_size)
        # attns: batch size x bean size x out length x inp length
        attns = torch.stack(
                [attn.view(batch_size, -1, attn.size(1)) for attn in attn_list],
                dim=2)
        return finished, attns
    return finished
