import collections

import numpy as np
import torch
from torch.autograd import Variable

from program_synthesis.datasets import data, executor
from program_synthesis.datasets.karel import karel_runtime, parser_for_synthesis
from program_synthesis.models import (
    base,
    beam_search,
    karel_model,
    prepare_spec,
)
from program_synthesis.models.modules import karel, karel_trace


class TracePredictionModel(karel_model.BaseKarelModel):
    def __init__(self, args):
        self.model = karel_trace.TracePrediction(args)
        self.kr = karel_runtime.KarelRuntime()

        super(TracePredictionModel, self).__init__(args)

    def compute_loss(self, batch):
        (input_grids, output_grids, trace_grids, input_actions, output_actions,
         io_embed_indices, _) = batch

        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            trace_grids = trace_grids.cuda(async=True)
            input_actions = input_actions.cuda(async=True)
            output_actions = output_actions.cuda(async=True)
            io_embed_indices = io_embed_indices.cuda(async=True)

        io_embed = self.model.encode(input_grids, output_grids)
        logits, labels = self.model.decode(
            io_embed, trace_grids, input_actions,
            output_actions, io_embed_indices)
        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def inference(self, batch):
        input_grids = batch.input_grids
        output_grids = batch.output_grids
        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)

        io_embed = self.model.encode(input_grids, output_grids)
        init_state = karel_trace.TraceDecoderState(
            # batch.input_grids is always on CPU, unlike input_grids
            batch.input_grids.data.numpy().astype(bool),
            *self.model.decoder.init_state(input_grids.shape[0]))
        memory = karel.LGRLMemory(io_embed)

        sequences = beam_search.beam_search(
            len(input_grids),
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        sequences = [[[karel_trace.id_to_action[i] for i in seq]
                      for seq in item_sequences]
                     for item_sequences in sequences]
        return [
            base.InferenceResult(
                code_sequence=item_sequences[0],
                info={'trees_checked': 1,
                      'candidates': item_sequences})
            for item_sequences in sequences
        ]

    def debug(self, batch):
        pass

    def eval(self, batch):
        results = self.inference(batch)
        correct = 0

        input_grids = batch.input_grids.data.numpy().astype(bool)
        output_grids = batch.output_grids.data.numpy().astype(bool)

        for i, result in enumerate(results):
            action_seq = result.code_sequence
            self.kr.init_from_array(input_grids[i])
            for action in action_seq:
                getattr(self.kr, action)()
            correct += np.all(input_grids[i] == output_grids[i])

        return {'correct': correct, 'total': len(results)}

    def batch_processor(self, for_eval):
        return TracePredictionBatchProcessor(for_eval)


TracePredictionExample = collections.namedtuple(
    'TracePredictionExample',
    ('input_grids', 'output_grids', 'trace_grids', 'input_actions',
     'output_actions', 'io_embed_indices', 'code'))


class TracePredictionBatchProcessor(object):

    def __init__(self, for_eval):
        self.for_eval = for_eval
        self.executor = executor.KarelExecutor()

    def __call__(self, batch):
        input_grids, output_grids = [
            torch.zeros(
                sum(len(item.input_tests) for item in batch), 15, 18, 18)
            for _ in range(2)
        ]
        trace_grids = [[] for _ in range(input_grids.shape[0])]
        # 0 for <s>
        actions = [[0] for _ in range(input_grids.shape[0])]

        # grids: starting grid, after action 1, after action 2
        # input actions: <s>, action 1, action 2
        # output actions: action 1, action 2, </s>
        idx = 0
        for item in batch:
            code = item.code_sequence
            for test in item.input_tests:
                input_grids[idx].view(-1)[test['input']] = 1
                output_grids[idx].view(-1)[test['output']] = 1
                result, trace = self.executor.execute(
                    code, None, test['input'], record_trace=True)
                # TODO: Eliminate need to use ravel
                for grid in trace.grids:
                    field = np.zeros((15, 18, 18), dtype=np.bool)
                    field.ravel()[grid] = 1
                    trace_grids[idx].append(field)
                for event in trace.events:
                    action_id = karel_trace.action_to_id.get(event.type, -1)
                    if action_id == -1:
                        continue
                    actions[idx].append(action_id)
                # 1: </s>
                actions[idx].append(1)
                assert len(trace_grids[idx]) == len(actions[idx]) - 1
                idx += 1

        input_grids, output_grids = [
            Variable(t) for t in (input_grids, output_grids)
        ]
        trace_grids = karel_model.lists_to_packed_sequence(
            trace_grids, (15, 18, 18), torch.FloatTensor,
            lambda item, batch_idx, out: out.copy_(torch.from_numpy(item.astype(np.float32))))
        input_actions = prepare_spec.lists_to_packed_sequence(
            [lst[:-1] for lst in actions],
            lambda x: x,
            cuda=False,
            volatile=False)
        output_actions = prepare_spec.lists_to_packed_sequence(
            [lst[1:] for lst in actions],
            lambda x: x,
            cuda=False,
            volatile=False)
        io_embed_indices = torch.LongTensor([
            idx
            for b in input_actions.ps.batch_sizes
            for idx in input_actions.orig_to_sort[:b]
        ])

        code = (
            [item.code_sequence for item in batch for _ in item.input_tests]
            if self.for_eval else None)

        return TracePredictionExample(input_grids, output_grids, trace_grids,
                input_actions, output_actions, io_embed_indices, code)


class CodeFromTracesModel(karel_model.BaseKarelModel):
    def __init__(self, args):
        self.args = args
        self.vocab = data.PlaceholderVocab(
            data.load_vocab(args.word_vocab), self.args.num_placeholders)
        self.model = karel_trace.CodeFromTraces(
            len(self.vocab) - self.args.num_placeholders, args)
        self.executor = executor.get_executor(args)()

        self.trace_grid_lengths = []
        self.trace_event_lengths  = []
        self.trace_lengths = []
        super(CodeFromTracesModel, self).__init__(args)

    def compute_loss(self, batch):
        if self.args.cuda:
            batch = batch.cuda_train()

        io_embed, trace_memory = self.model.encode(
                batch.input_grids,
                batch.output_grids,
                batch.trace_grids,
                batch.conds,
                batch.actions,
                batch.interleave)

        logits, labels = self.model.decode(
                batch.batch_size,
                batch.pairs_per_example,
                io_embed,
                trace_memory,
                batch.input_code,
                batch.output_code)

        return self.criterion(
            logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1))

    def debug(self, batch):
        pass

    def inference(self, batch):
        if self.args.cuda:
            batch = batch.cuda_infer()

        io_embed, trace_memory = self.model.encode(
                batch.input_grids,
                batch.output_grids,
                batch.trace_grids,
                batch.conds,
                batch.actions,
                batch.interleave)

        init_state = self.model.decoder.init_state(
                batch.batch_size, batch.pairs_per_example)
        memory = self.model.decoder.prepare_memory(
                batch.batch_size, batch.pairs_per_example, io_embed,
                trace_memory)

        sequences = beam_search.beam_search(
            batch.batch_size,
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        return self._try_sequences(self.vocab, sequences, batch.input_grids,
                                   batch.output_grids, self.args.max_beam_trees)

    def eval(self, batch):
        correct = 0
        for res, example in zip(self.inference(batch), batch.orig_examples):
            stats = executor.evaluate_code(
                res.code_sequence, None, example.tests, self.executor.execute)
            if stats['correct'] == stats['total']:
                correct += 1
        return {'correct': correct, 'total': len(batch.orig_examples)}


    def batch_processor(self, for_eval):
        return CodeFromTracesBatchProcessor(self.vocab, for_eval)


class CodeFromTracesExample(
        collections.namedtuple('CodeFromTracesExample', (
            'batch_size', 'pairs_per_example',
            'input_grids', 'output_grids', 'trace_grids', 'conds',
            'actions', 'interleave', 'input_code', 'output_code', 'orig_examples'))):

    def cuda_train(self):
        self = self.cuda_infer()
        input_code = self.input_code.cuda(async=True)
        output_code = self.output_code.cuda(async=True)
        return CodeFromTracesExample(
                self.batch_size,
                self.pairs_per_example,
                self.input_grids, self.output_grids, self.trace_grids,
                self.conds, self.actions,
                self.interleave, input_code, output_code, self.orig_examples)

    def cuda_infer(self):
        input_grids = self.input_grids.cuda(async=True)
        output_grids = self.output_grids.cuda(async=True)
        trace_grids = self.trace_grids.cuda(async=True)
        conds = self.conds.cuda(async=True)
        actions = self.actions.cuda(async=True)

        return CodeFromTracesExample(
                self.batch_size,
                self.pairs_per_example,
                input_grids, output_grids, trace_grids, conds, actions,
                self.interleave, self.input_code, self.output_code,
                self.orig_examples)


class CodeFromTracesBatchProcessor(object):
    def __init__(self, vocab, for_eval):
        self.vocab = vocab
        self.for_eval = for_eval
        self.karel_parser = parser_for_synthesis.KarelForSynthesisParser()
        self.karel_parser.karel.action_callback = self.action_callback

        self.full_grid =  np.zeros((15, 18, 18), dtype=np.bool)
        self.actions = None
        self.grids = None
        self.conds = None

    def __call__(self, batch):
        batch_size = len(batch)
        pairs_per_example = len(batch[0].input_tests)

        input_grids, output_grids, code_seqs = karel_model.encode_grids_and_outputs(
            batch, self.vocab)

        trace_grids = []
        conds = []
        actions = []
        interleave = []
        code_seqs = []

        for batch_idx, item in enumerate(batch):
            prog = self.karel_parser.parse(item.code_sequence)
            code_seqs.append(['<s>'] + item.code_sequence + ['</s>'])

            for test_idx, test in enumerate(item.input_tests):
                self.full_grid[:] = False
                self.full_grid.ravel()[test['input']] = True
                self.karel_parser.karel.init_from_array(self.full_grid)
                self.reset_tracer()

                prog()
                self.actions.append(1) # </s>

                trace_grids.append(self.grids)
                conds.append(self.conds)
                actions.append(self.actions)
                interleave.append([0] + [1, 0] * (len(self.grids) - 1))

        trace_grids = karel_model.lists_to_packed_sequence(
            trace_grids, (15, 18, 18), torch.FloatTensor,
            lambda item, batch_idx, out: out.copy_(torch.from_numpy(item.astype(np.float32))))

        conds = karel_model.lists_to_packed_sequence(
            conds, (4,), torch.LongTensor,
            lambda item, batch_idx, out: out.copy_(torch.LongTensor(item)))

        actions = prepare_spec.lists_to_packed_sequence(
            actions,
            lambda x: x,
            cuda=False,
            volatile=False)

        interleave = prepare_spec.prepare_interleave_packed_sequences(
                (trace_grids, actions), interleave)

        input_code = prepare_spec.lists_to_packed_sequence(
                [seq[:-1] for seq in code_seqs],
                self.vocab.stoi,
                cuda=False,
                volatile=False)

        output_code = prepare_spec.lists_to_packed_sequence(
                [seq[1:] for seq in code_seqs],
                self.vocab.stoi,
                cuda=False,
                volatile=False)

        orig_examples = batch if self.for_eval else None

        return CodeFromTracesExample(
                batch_size,
                pairs_per_example,
                input_grids,
                output_grids,
                trace_grids,
                conds,
                actions,
                interleave,
                input_code,
                output_code,
                orig_examples)

    def reset_tracer(self):
        self.actions = []
        self.grids = []
        self.conds = []
        self.record_grid_state()

    def action_callback(self, action_name, success, span):
        self.actions.append(karel_trace.action_to_id[action_name])
        self.record_grid_state()

    def record_grid_state(self):
        runtime = self.karel_parser.karel
        grid = self.full_grid.copy()
        cond = [
            int(v)
            for v in (runtime.frontIsClear(), runtime.leftIsClear(),
                      runtime.rightIsClear(), runtime.markersPresent())
        ]
        self.grids.append(grid)
        self.conds.append(cond)
