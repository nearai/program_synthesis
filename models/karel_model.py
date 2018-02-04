import collections
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from base import BaseCodeModel, InferenceResult
from datasets import data, dataset, executor
from modules import karel
import beam_search
import prepare_spec


def code_to_tokens(seq, vocab):
    tokens = []
    for i in seq:
        if i == 1:  # </s>
            break
        tokens.append(vocab.itos(i))
    return tokens


def encode_grids_and_outputs(batch, vocab):
    # TODO: Don't hard-code 5 I/O examples
    input_grids, output_grids = [
        torch.zeros(len(batch), 5, 15, 18, 18) for _ in range(2)
    ]
    for batch_idx, item in enumerate(batch):
        assert len(item.input_tests) == 5
        for test_idx, test in enumerate(item.input_tests):
            inp, out = test['input'], test['output']
            input_grids[batch_idx, test_idx].view(-1)[inp] = 1
            output_grids[batch_idx, test_idx].view(-1)[out] = 1
    input_grids, output_grids = [
        Variable(t) for t in (input_grids, output_grids)
    ]
    code_seqs = prepare_spec.lists_padding_to_tensor(
        [item.code_sequence for item in batch], vocab.stoi, False)
    return input_grids, output_grids, code_seqs


def maybe_cuda(tensor, async=False):
    if tensor is None:
        return None
    return tensor.cuda(async=async)


class BaseKarelModel(BaseCodeModel):
    def eval(self, batch):
        results = self.inference(batch)
        correct = 0
        code_seqs = batch.code_seqs.cpu()
        for code_seq, res in zip(code_seqs, results):
            code_tokens = code_to_tokens(code_seq.data[1:], self.vocab)
            if code_tokens == res.code_sequence:
                correct += 1
        return {'correct': correct, 'total': len(code_seqs)}

    def _try_sequences(self, vocab, sequences, input_grids, output_grids,
                       beam_size):
        result = [[] for _ in range(len(sequences))]
        counters = [0 for _ in range(len(sequences))]
        candidates = [[] for _ in range(len(sequences))]
        max_eval_trials = self.args.max_eval_trials or beam_size
        for batch_id, outputs in enumerate(sequences):
            input_tests = [
                {
                    'input': np.where(inp.numpy().ravel())[0].tolist(),
                    'output': np.where(out.numpy().ravel())[0].tolist(),
                }
                for inp, out in zip(
                    torch.split(input_grids[batch_id].data.cpu(), 1),
                    torch.split(output_grids[batch_id].data.cpu(), 1), )
            ]
            candidates[batch_id] = [[vocab.itos(idx) for idx in ids]
                                    for ids in outputs]
            for code in candidates[batch_id][:max_eval_trials]:
                counters[batch_id] += 1
                stats = executor.evaluate_code(code, None, input_tests,
                                               self.executor.execute)
                ok = (stats['correct'] == stats['total'])
                if ok:
                    result[batch_id] = code
                    break
        return [
            InferenceResult(
                code_sequence=seq,
                info={'trees_checked': c,
                      'candidates': cand})
            for seq, c, cand in zip(result, counters, candidates)
        ]


class KarelLGRLModel(BaseKarelModel):
    def __init__(self, args):
        self.args = args
        self.vocab = data.PlaceholderVocab(
            data.load_vocab(args.word_vocab), self.args.num_placeholders)
        self.model = karel.LGRLKarel(
            len(self.vocab) - self.args.num_placeholders, args)
        self.executor = executor.get_executor(args)()
        super(KarelLGRLModel, self).__init__(args)

    def compute_loss(self, (input_grids, output_grids, code_seqs, _)):
        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            code_seqs = code_seqs.cuda(async=True)
        # io_embeds shape: batch size x num pairs (5) x hidden size (512)
        io_embed = self.model.encode(input_grids, output_grids)
        logits, labels = self.model.decode(io_embed, code_seqs)
        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def debug(self, batch):
        batch = [x[0:1] for x in batch]
        code = code_to_tokens(batch.code_seqs.data[0, 1:], self.vocab)
        print("Code: %s" % ' '.join(code))
        res, = self.inference(batch)
        print("Out:  %s" % ' '.join(res.code_sequence))

    def inference(self, (input_grids, output_grids, _1, _2)):
        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)

        io_embed = self.model.encode(input_grids, output_grids)
        init_state = karel.LGRLDecoderState(*self.model.decoder.zero_state(
            io_embed.shape[0], io_embed.shape[1]))
        memory = karel.LGRLMemory(io_embed)

        sequences = beam_search.beam_search(
            len(input_grids),
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        return self._try_sequences(self.vocab, sequences, input_grids,
                                   output_grids, self.args.max_beam_trees)

    def batch_processor(self, for_eval):
        return KarelLGRLBatchProcessor(self.vocab, for_eval)


KarelLGRLExample = collections.namedtuple(
    'KarelLGRLExample',
    ('input_grids', 'output_grids', 'code_seqs', 'orig_examples'))


class KarelLGRLBatchProcessor(object):
    def __init__(self, vocab, for_eval):
        self.vocab = vocab
        self.for_eval = for_eval

    def __call__(self, batch):
        input_grids, output_grids, code_seqs = encode_grids_and_outputs(
            batch, self.vocab)
        orig_examples = batch if self.for_eval else None
        return KarelLGRLExample(input_grids, output_grids, code_seqs,
                                orig_examples)


class KarelLGRLRefineModel(BaseKarelModel):
    def __init__(self, args):
        self.args = args
        self.vocab = data.PlaceholderVocab(
            data.load_vocab(args.word_vocab), self.args.num_placeholders)
        self.model = karel.LGRLRefineKarel(
            len(self.vocab) - self.args.num_placeholders, args)
        self.executor = executor.get_executor(args)()
        super(KarelLGRLRefineModel, self).__init__(args)

    def compute_loss(self, (input_grids, output_grids, code_seqs, ref_code,
                            ref_trace_grids, ref_trace_events, _)):
        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            code_seqs = code_seqs.cuda(async=True)
            ref_code = ref_code.cuda(async=True)
            ref_trace_grids = ref_trace_grids.cuda(async=True)
            ref_trace_events = ref_trace_events.cuda(async=True)

        # io_embeds shape: batch size x num pairs (5) x hidden size (512)
        io_embed, ref_code_memory, ref_trace_memory = self.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events)
        logits, labels = self.model.decode(io_embed, ref_code_memory,
                                           ref_trace_memory, code_seqs)
        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def debug(self, batch):
        code = code_to_tokens(batch.code_seqs.data[0, 1:], self.vocab)
        print("Code: %s" % ' '.join(code))

    def inference(self, (input_grids, output_grids, _1, ref_code,
                         ref_trace_grids, ref_trace_events, _2)):
        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            ref_code = maybe_cuda(ref_code, async=True)
            ref_trace_grids = maybe_cuda(ref_trace_grids, async=True)
            ref_trace_events = maybe_cuda(ref_trace_events, async=True)

        io_embed, ref_code_memory, ref_trace_memory = self.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events)
        init_state = self.model.decoder.zero_state(io_embed.shape[0],
                                                   io_embed.shape[1])
        memory = self.model.decoder.prepare_memory(io_embed, ref_code_memory,
                                                   ref_trace_memory)

        sequences = beam_search.beam_search(
            len(input_grids),
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        return self._try_sequences(self.vocab, sequences, input_grids,
                                   output_grids, self.args.max_beam_trees)

    def batch_processor(self, for_eval):
        return KarelLGRLRefineBatchProcessor(self.args, self.vocab, for_eval)


KarelLGRLRefineExample = collections.namedtuple('KarelLGRLRefineExample', (
    'input_grids', 'output_grids', 'code_seqs', 'ref_code', 'ref_trace_grids',
    'ref_trace_events', 'orig_examples'))


class KarelLGRLRefineBatchProcessor(object):
    def __init__(self, args, vocab, for_eval):
        self.args = args
        self.vocab = vocab
        self.for_eval = for_eval

    def __call__(self, batch):
        input_grids, output_grids, code_seqs = encode_grids_and_outputs(
            batch, self.vocab)

        if self.args.karel_code_enc == 'none':
            ref_code = None
        else:
            ref_code = prepare_spec.lists_to_packed_sequence(
                [item.ref_example.code_sequence for item in batch],
                self.vocab.stoi,
                False,
                volatile=False)

        if self.args.karel_trace_enc == 'none':
            ref_trace_grids, ref_trace_events = None, None
        else:
            ref_trace_grids, ref_trace_events = self.prepare_traces(batch)

        orig_examples = batch if self.for_eval else None

        return KarelLGRLRefineExample(input_grids, output_grids, code_seqs,
                                      ref_code, ref_trace_grids,
                                      ref_trace_events, orig_examples)

    def prepare_traces(self, batch):
        ref_trace_grids = torch.zeros(
            sum(
                len(test['trace'].grids)
                for item in batch
                for test in item.ref_example.input_tests), 15, 18, 18)
        trace_grids_lists, sort_to_orig = prepare_spec.sort_lists_by_length([
            test['trace'].grids
            for item in batch for test in item.ref_example.input_tests
        ])
        lengths = prepare_spec.lengths(trace_grids_lists)
        batch_bounds = prepare_spec.batch_bounds_for_packing(lengths)
        idx = 0

        last_grids = [set() for _ in trace_grids_lists]
        for i, bound in enumerate(batch_bounds):
            for batch_idx, trace_grids in enumerate(trace_grids_lists[:bound]):
                if isinstance(trace_grids[i], dict):
                    last_grid = last_grids[batch_idx]
                    #assert last_grid.isdisjoint(trace_grids[i]['plus'])
                    #assert last_grid >= trace_grids[i]['minus']
                    last_grid.update(trace_grids[i]['plus'])
                    last_grid.difference_update(trace_grids[i]['minus'])
                else:
                    last_grid = last_grids[batch_idx] = set(trace_grids[i])
                ref_trace_grids[idx].view(-1)[list(last_grid)] = 1
                idx += 1
        ref_trace_grids = Variable(ref_trace_grids)

        ref_trace_grids = prepare_spec.PackedSequencePlus(
            nn.utils.rnn.PackedSequence(ref_trace_grids, batch_bounds),
            lengths, sort_to_orig)
        # TODO: replace this placeholder
        ref_trace_events = None
        return ref_trace_grids, ref_trace_events
