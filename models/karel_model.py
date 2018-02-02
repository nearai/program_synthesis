import torch
from torch import nn
from torch.autograd import Variable

from base import BaseCodeModel, InferenceResult
from datasets import data
from datasets import executor
from modules import karel
import beam_search
import prepare_spec


class KarelLGRLModel(BaseCodeModel):
    def __init__(self, args):
        self.args = args
        self.vocab = data.load_vocab(args.word_vocab)
        self.model = karel.LGRLKarel(len(self.vocab), args)
        self.executor = executor.get_executor(args)()
        super(KarelLGRLModel, self).__init__(args)

    def compute_loss(self, batch):
        vocab = self.reset_vocab()
        logits, labels = self.encode_decode(vocab, batch)
        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def encode(self, vocab, batch):
        # TODO: Don't hard-code 5 I/O examples
        input_grids, output_grids = [
            torch.zeros(
                len(batch) * 5,
                15,
                18,
                18,
                out=torch.cuda.FloatTensor()
                if self.args.cuda else torch.FloatTensor()) for _ in range(2)
        ]
        idx = 0
        for item in batch:
            for test in item.input_tests:
                inp, out = test['input'], test['output']
                input_grids[idx].view(-1)[inp] = 1
                output_grids[idx].view(-1)[out] = 1
                idx += 1
        input_grids, output_grids = [
            Variable(t) for t in (input_grids, output_grids)
        ]
        return self.model.encode(input_grids, output_grids).view(
            len(batch), 5, -1)

    def encode_decode(self, vocab, batch):
        # io_embeds shape: batch size x num pairs (5) x hidden size (512)
        io_embed = self.encode(vocab, batch)
        outputs = prepare_spec.lists_padding_to_tensor(
            [item.code_sequence for item in batch], vocab.stoi, self.args.cuda)

        logits, labels = self.model.decode(io_embed, outputs)
        return logits, labels

    def debug(self, batch):
        item = batch[0]
        print("Code: %s" % ' '.join(item.code_sequence))
        res, = self.inference([item])
        print("Out:  %s" % ' '.join(res.code_sequence))

    def inference(self, batch):
        vocab = self.reset_vocab()
        io_embed = self.encode(vocab, batch)
        init_state = karel.LGRLDecoderState(*self.model.decoder.zero_state(
            io_embed.shape[0], io_embed.shape[1]))
        memory = karel.LGRLMemory(io_embed)

        sequences = beam_search.beam_search(
            len(batch),
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        return self._try_sequences(vocab, sequences, batch,
                                   self.args.max_beam_trees)


class KarelLGRLRefineModel(BaseCodeModel):
    def __init__(self, args):
        self.args = args
        self.vocab = data.load_vocab(args.word_vocab)
        self.model = karel.LGRLRefineKarel(len(self.vocab), args)
        self.executor = executor.get_executor(args)()
        super(KarelLGRLRefineModel, self).__init__(args)

    def compute_loss(self, batch):
        vocab = self.reset_vocab()
        logits, labels = self.encode_decode(vocab, batch)
        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def encode(self, vocab, batch):
        # TODO: Don't hard-code 5 I/O examples
        input_grids, output_grids = [
            torch.zeros(
                len(batch) * 5,
                15,
                18,
                18,
                out=torch.cuda.FloatTensor()
                if self.args.cuda else torch.FloatTensor()) for _ in range(2)
        ]
        idx = 0
        for item in batch:
            for test in item.input_tests:
                inp, out = test['input'], test['output']
                input_grids[idx].view(-1)[inp] = 1
                output_grids[idx].view(-1)[out] = 1
                idx += 1
        input_grids, output_grids = [
            Variable(t) for t in (input_grids, output_grids)
        ]

        if self.args.karel_code_enc == 'none':
            ref_code = None
        else:
            ref_code = prepare_spec.lists_to_packed_sequence(
                [item.ref_example.code_sequence for item in batch],
                vocab.stoi,
                self.args.cuda,
                volatile=False)

        if self.args.karel_trace_enc == 'none':
            ref_trace = None
        else:
            ref_trace = self.prepare_traces(batch)

        io_embed, c, t =  self.model.encode(
                input_grids, output_grids, ref_code, ref_trace)
        return io_embed.view(len(batch), 5, -1), c, t

    def prepare_traces(self, batch):
        ref_trace_grids = torch.zeros(
            sum(
                len(test['trace'].grids)
                for item in batch for test in item.ref_example.input_tests),
            15,
            18,
            18,
            out=torch.cuda.FloatTensor()
            if self.args.cuda else torch.FloatTensor())
        trace_grids_lists, sort_to_orig = prepare_spec.sort_lists_by_length([
            test['trace'].grids
            for item in batch for test in item.ref_example.input_tests
        ])
        lengths = prepare_spec.lengths(trace_grids_lists)
        batch_bounds = prepare_spec.batch_bounds_for_packing(lengths)
        idx = 0

        last_grids = [set() for _ in trace_grids_lists]
        for i, bound in enumerate(
                prepare_spec.batch_bounds_for_packing(
                    prepare_spec.lengths(trace_grids_lists))):
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

        return executor.KarelTrace(
            grids=prepare_spec.PackedSequencePlus(
                nn.utils.rnn.PackedSequence(ref_trace_grids, batch_bounds),
                lengths, sort_to_orig),
            events=None)

    def encode_decode(self, vocab, batch):
        # io_embeds shape: batch size x num pairs (5) x hidden size (512)
        io_embed, ref_code_memory, ref_trace_memory = self.encode(vocab, batch)
        outputs = prepare_spec.lists_padding_to_tensor(
            [item.code_sequence for item in batch], vocab.stoi, self.args.cuda)

        logits, labels = self.model.decode(io_embed, ref_code_memory,
                                           ref_trace_memory, outputs)
        return logits, labels

    def debug(self, batch):
        item = batch[0]
        print("Code: %s" % ' '.join(item.code_sequence))
        if item.ref_example:
            print("Ref:  %s" % ' '.join(item.ref_example.code_sequence))
        res, = self.inference([item])
        print("Out:  %s" % ' '.join(res.code_sequence))

    def inference(self, batch):
        vocab = self.reset_vocab()
        io_embed, ref_code_memory, ref_trace_memory = self.encode(vocab, batch)
        init_state = self.model.decoder.zero_state(io_embed.shape[0],
                                                   io_embed.shape[1])
        memory = self.model.decoder.prepare_memory(
                io_embed, ref_code_memory, ref_trace_memory)

        sequences = beam_search.beam_search(
            len(batch),
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        return self._try_sequences(vocab, sequences, batch,
                                   self.args.max_beam_trees)
