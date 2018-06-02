import sys
import os
import math
import time
import collections

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F


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
                max_decoder_length,
                cuda=False,
                return_attention=False,
                return_beam_search_result=False,
                volatile=True):
    # enc: batch size x hidden size
    # memory: batch size x sequence length x hidden size
    tt = torch.cuda if cuda else torch
    prev_tokens = Variable(tt.LongTensor(batch_size).fill_(0), volatile=volatile)
    prev_acc_log_probs = Variable(tt.FloatTensor(batch_size, 1).fill_(0), volatile=volatile)
    prev_hidden = enc
    finished_beams = [[] for _ in range(batch_size)]
    unfinished_beams = [[BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0)
                         for _ in range(beam_size)] for _ in range(batch_size)]
    seq_is_finished = [False] * batch_size
    # seq_indices: 0, ..., 0, 1, ..., 1, ..., b, ..., b
    # where b is the batch size, and each group of numbers has as many elements
    # as the beam size.
    seq_indices = Variable(
        torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, beam_size).view(-1),
        volatile=volatile)

    expanded_masked_memory = masked_memory.expand_by_beam(beam_size)

    attn_list = [] if return_attention else None
    current_beam_size = 1
    for step in range(max_decoder_length):
        if current_beam_size == 1:
            current_memory = masked_memory
        elif current_beam_size == beam_size:
            current_memory = expanded_masked_memory
        else:
            current_memory = masked_memory.expand_by_beam(current_beam_size)
        hidden, logits = decode_fn(prev_tokens.view(-1), prev_hidden,
                                   current_memory, attentions=attn_list)

        vocab_size = logits.size(1)
        # log_probs: batch size x beam size x vocab size
        log_probs = F.log_softmax(logits, dim=-1).view(batch_size, -1, vocab_size)
        acc_log_probs = log_probs + prev_acc_log_probs.unsqueeze(2)
        # log_probs_flat: batch size x beam_size * vocab_size
        acc_log_probs_flat = acc_log_probs.view(batch_size, -1)
        current_beam_size = min(acc_log_probs_flat.size(1), beam_size)
        # indices: batch size x beam size
        # Each entry is in [0, beam_size * vocab_size)
        prev_acc_log_probs, indices_flat = acc_log_probs_flat.topk(current_beam_size, dim=1)
        # prev_tokens: batch_size x beam size
        # Each entry indicates which token should be added to each beam.
        prev_tokens = indices_flat % vocab_size
        # This takes a lot of time... about 50% of the whole thing.
        indices_flat = indices_flat.cpu()
        # beam_indices: batch size x beam size
        # Each entry is in [0, beam_size), indicating which previous beam to extend.
        beam_indices = indices_flat / vocab_size
        if current_beam_size == beam_size:
            seq_beam_indices = torch.stack([seq_indices, beam_indices.view(-1)])
        else:
            cur_seq_indices = Variable(
                torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(
                    1, current_beam_size).view(-1), volatile=volatile)
            seq_beam_indices = torch.stack([cur_seq_indices, beam_indices.view(-1)])
        # prev_hidden: (batch size * beam size) x hidden size
        # Contains the hidden states which produced the top-k (k = beam size)
        # tokens, and should be extended in the step.
        prev_hidden = hidden.select_for_beams(batch_size, seq_beam_indices)

        prev_unfinished_beams = unfinished_beams
        unfinished_beams = [[] for _ in range(batch_size)]
        can_stop = True
        prev_acc_log_probs_np = prev_acc_log_probs.data.cpu().numpy()
        log_probs_np = log_probs.data.cpu().numpy()
        beam_indices_np = beam_indices.data.numpy()
        prev_tokens_np = prev_tokens.data.cpu().numpy()
        for seq_id in range(batch_size):
            if seq_is_finished[seq_id]:
                continue
            # print(step, finished_beams[batch_id])
            if len(finished_beams[seq_id]) >= beam_size:
                # If last in finished has bigger log prob then best in topk, stop.
                if finished_beams[seq_id][-1].total_log_prob > prev_acc_log_probs_np[seq_id, 0]:
                    seq_is_finished[seq_id] = True
                    continue
            for beam_id in range(current_beam_size):
                token = prev_tokens_np[seq_id, beam_id]
                new_beam_id = beam_indices_np[seq_id, beam_id]
                # print(step, batch_id, idx, 'token', token, kidx, 'prev', prev_result[batch_id][kidx], prev_probs.data[batch_id][idx])
                if token == 1:  # 1 == </S>
                    finished_beams[seq_id].append(BeamSearchResult(
                        sequence=prev_unfinished_beams[seq_id][new_beam_id].sequence,
                        total_log_prob=prev_acc_log_probs_np[seq_id, beam_id],
                        log_probs=prev_unfinished_beams[seq_id][new_beam_id].log_probs +
                                  [log_probs_np[seq_id, new_beam_id, token]]))
                    unfinished_beams[seq_id].append(BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0))
                    prev_acc_log_probs.data[seq_id, beam_id] = float('-inf')
                else:
                    unfinished_beams[seq_id].append(BeamSearchResult(
                        sequence=prev_unfinished_beams[seq_id][new_beam_id].sequence + [token],
                        total_log_prob=prev_acc_log_probs_np[seq_id, beam_id],
                        log_probs=prev_unfinished_beams[seq_id][new_beam_id].log_probs +
                                  [log_probs_np[seq_id, new_beam_id, token]]))
                    can_stop = False
            if len(finished_beams[seq_id]) >= beam_size:
                # Sort and clip.
                finished_beams[seq_id] = sorted(
                    finished_beams[seq_id], key=lambda x: x.total_log_prob, reverse=True)[:beam_size]
        if can_stop:
            break

    for seq_id in range(batch_size):
        # If there is deficit in finished, fill it in with highest probable results.
        if len(finished_beams[seq_id]) < beam_size:
            i = 0
            while i < beam_size and len(finished_beams[seq_id]) < beam_size:
                if unfinished_beams[seq_id][i]:
                    finished_beams[seq_id].append(unfinished_beams[seq_id][i])
                i += 1

    if not return_beam_search_result:
        for seq_id in range(batch_size):
            finished_beams[seq_id] = [x.sequence for x in finished_beams[seq_id]]

    if return_attention:
        # all elements of attn_list: (batch size * beam size) x input length
        attn_list[0] = expand_by_beam(attn_list[0], beam_size)
        # attns: batch size x bean size x out length x inp length
        attns = torch.stack(
                [attn.view(batch_size, -1, attn.size(1)) for attn in attn_list],
                dim=2)
        return finished_beams, attns
    return finished_beams
