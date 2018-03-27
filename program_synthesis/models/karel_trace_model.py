import collections
import copy

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
from program_synthesis.models.modules import karel, karel_trace, karel_common


class KarelTracer(object):

    def __init__(self, runtime, keep_failures=True):
        self.runtime = runtime
        self.runtime.action_callback = self.action_callback
        self.full_grid =  np.zeros((15, 18, 18), dtype=np.bool)
        self.keep_failures = keep_failures

    def reset(self, indices=None, grid=None):
        assert (indices is not None) ^ (grid is not None)
        if indices is not None:
            self.full_grid[:] = False
            self.full_grid.ravel()[indices] = True
        elif grid is not None:
            self.full_grid[:] = grid
        self.runtime.init_from_array(self.full_grid)

        self.actions = []
        self.grids = []
        self.conds = []
        self.record_grid_state()

    def action_callback(self, action_name, success, span):
        if success or self.keep_failures:
            self.actions.append(action_name)
            self.record_grid_state()

    def record_grid_state(self):
        runtime = self.runtime
        grid = self.full_grid.copy()
        cond = executor.KarelCondValues(*[
            int(v)
            for v in (runtime.frontIsClear(), runtime.leftIsClear(),
                      runtime.rightIsClear(), runtime.markersPresent())
        ])
        self.grids.append(grid)
        self.conds.append(cond)


class TracePredictionModel(karel_model.BaseKarelModel):
    def __init__(self, args):
        self.model = karel_trace.TracePrediction(args)
        self.kr = karel_runtime.KarelRuntime()
        self.tracer = KarelTracer(self.kr, keep_failures=False)
        self.eval_kr = karel_runtime.KarelRuntime()

        self.all_infer = 0
        self.correct_infer = 0

        super(TracePredictionModel, self).__init__(args)

    def compute_loss(self, batch):
        (input_grids, output_grids, trace_grids, conds, input_actions,
         output_actions, io_embed_indices, _) = batch

        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            trace_grids = trace_grids.cuda(async=True)
            conds = conds.cuda(async=True)
            input_actions = input_actions.cuda(async=True)
            output_actions = output_actions.cuda(async=True)
            io_embed_indices = io_embed_indices.cuda(async=True)

        io_embed = self.model.encode(input_grids, output_grids)
        logits, labels = self.model.decode(
            io_embed, trace_grids,  conds, input_actions,
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
        # TODO: Make InferenceResult more general so that we don't have to shove
        # an action sequence into "code_sequence".
        return [
            base.InferenceResult(
                code_sequence=item_sequences[0],
                info={'trees_checked': 1,
                      'candidates': item_sequences})
            for item_sequences in sequences
        ]

    def process_infer_results(self, batch, inference_results,
            counters=collections.Counter()):
        grid_idx = 0

        input_grids = batch.input_grids.data.numpy().astype(bool)
        output_grids = batch.output_grids.data.numpy().astype(bool)

        output = []
        for orig_example in batch.orig_examples:
            orig_example = copy.deepcopy(orig_example)
            output.append(orig_example)
            for test in (orig_example.input_tests + orig_example.tests
                         if self.args.karel_trace_inc_val else []):
                result = inference_results[grid_idx]
                candidates = result.info['candidates']
                if self.args.max_eval_trials:
                    candidates = candidates[:self.args.max_eval_trials]

                found = False
                for cand in candidates:
                    self.tracer.reset(grid=input_grids[grid_idx])
                    success = True
                    for action in cand:
                        success = getattr(self.kr, action)()
                        if not success:
                            counters['failure-' + action] += 1
                    if  np.all(
                            self.tracer.full_grid == output_grids[grid_idx]):
                        found = True
                        break

                counters['total'] += 1
                if found:
                    grids = self.tracer.grids
                    actions = self.tracer.actions
                    conds = self.tracer.conds
                    counters['correct'] += 1
                else:
                    grids = [input_grids[grid_idx], output_grids[grid_idx]]
                    actions = ['UNK']
                    # TODO fix conds
                    conds = [self.tracer.conds[0], self.tracer.conds[-1]]

                grids = karel_common.compress_trace(
                        [np.where(g.ravel())[0].tolist() for g in
                            grids])
                assert len(grids) >= 2
                test['trace'] = executor.KarelTrace(
                    grids=grids,
                    events=[
                        executor.KarelEvent(
                            timestep=t,
                            type=name,
                            span=None,
                            cond_span=None,
                            cond_value=None,
                            success=None)
                        for t, name in enumerate(actions)
                    ],
                    cond_values=conds)
                grid_idx += 1

        return [example.to_dict() for example in output]

    def debug(self, batch):
        pass

    def eval(self, batch):
        results = self.inference(batch)
        correct = 0

        input_grids = batch.input_grids.data.numpy().astype(bool)
        output_grids = batch.output_grids.data.numpy().astype(bool)

        for i, result in enumerate(results):
            action_seq = result.code_sequence
            self.eval_kr.init_from_array(input_grids[i])
            for action in action_seq:
                getattr(self.eval_kr, action)()
            # input_grids[i] is mutated in place by self.eval_kr.
            correct += np.all(input_grids[i] == output_grids[i])

        return {'correct': correct, 'total': len(results)}

    def batch_processor(self, for_eval):
        return TracePredictionBatchProcessor(self.args, for_eval)


TracePredictionExample = collections.namedtuple(
    'TracePredictionExample',
    ('input_grids', 'output_grids', 'trace_grids', 'conds', 'input_actions',
     'output_actions', 'io_embed_indices', 'orig_examples'))


class TracePredictionBatchProcessor(object):

    def __init__(self, args, for_eval):
        self.args = args
        self.for_eval = for_eval
        self.karel_parser = parser_for_synthesis.KarelForSynthesisParser()
        self.tracer = KarelTracer(self.karel_parser.karel)

    def __call__(self, batch):
        input_grids, output_grids = [
            torch.zeros(
                sum(
                    len(item.input_tests) + len(item.tests)
                    if self.args.karel_trace_inc_val else 0
                    for item in batch), 15, 18, 18) for _ in range(2)
        ]

        trace_grids = []
        conds = []
        actions = []

        # grids: starting grid, after action 1, after action 2
        # input actions: <s>, action 1, action 2
        # output actions: action 1, action 2, </s>
        idx = 0
        for item in batch:
            prog = self.karel_parser.parse(item.code_sequence)
            for test in (item.input_tests + item.tests
                         if self.args.karel_trace_inc_val else []):
                input_grids[idx].view(-1)[test['input']] = 1
                output_grids[idx].view(-1)[test['output']] = 1

                self.tracer.reset(indices=test['input'])
                self.tracer.actions.append('UNK') # means <s>
                prog()
                self.tracer.actions.append('</s>')  # </s>

                trace_grids.append(self.tracer.grids)
                conds.append(self.tracer.conds)
                actions.append([
                    karel_trace.action_to_id[name]
                    for name in self.tracer.actions
                ])
                assert len(trace_grids[-1]) == len(actions[-1]) - 1
                idx += 1

        input_grids, output_grids = [
            Variable(t) for t in (input_grids, output_grids)
        ]
        trace_grids = karel_model.lists_to_packed_sequence(
            trace_grids, (15, 18, 18), torch.FloatTensor,
            lambda item, batch_idx, out: out.copy_(torch.from_numpy(item.astype(np.float32))))
        conds = karel_model.lists_to_packed_sequence(
            conds, (4,), torch.LongTensor,
            lambda item, batch_idx, out: out.copy_(torch.LongTensor(item)))
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

        orig_examples = batch if self.for_eval else None

        return TracePredictionExample(input_grids, output_grids, trace_grids,
                                      conds, input_actions, output_actions,
                                      io_embed_indices, orig_examples)


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


class CodeFromTracesBatchProcessor(KarelTracer):
    def __init__(self, vocab, for_eval):
        self.vocab = vocab
        self.for_eval = for_eval
        self.karel_parser = parser_for_synthesis.KarelForSynthesisParser()
        self.tracer = KarelTracer(self.karel_parser.karel)

    def __call__(self, batch):
        batch_size = len(batch)
        pairs_per_example = len(batch[0].input_tests)

        input_grids, output_grids = karel_model.encode_io_grids(batch)
        code_seqs = [['<S>'] + item.code_sequence + ['</S>'] for item in batch]

        if 'trace' in batch[0].input_tests[0]:
            # TODO: Allow a mixture of tests with and without predefined traces
            assert all('trace' in test for item in batch for test in
                    item.input_tests)

            grids_lists = [
                test['trace'].grids
                for item in batch for test in item.input_tests
            ]
            trace_grids = karel_model.encode_trace_grids(grids_lists)

            conds = [
                test['trace'].cond_values
                for item in batch for test in item.input_tests
            ]

            def make_actions(test):
                actions_list = []
                for ev in test['trace'].events:
                    action_id = karel_trace.action_to_id.get(ev.type)
                    if action_id is not None:
                        actions_list.append(action_id)
                actions_list.append(1)   # </s>
                return actions_list
            actions = [
                make_actions(test) for item in batch
                for test in item.input_tests
            ]

            assert all(
                len(g) == len(a) for g, a in zip(grids_lists, actions))

            interleave = [[0] + [1, 0] * (len(grids_list) - 1)
                          for grids_list in grids_lists]
        else:
            trace_grids = []
            conds = []
            actions = []
            interleave = []

            for batch_idx, item in enumerate(batch):
                prog = self.karel_parser.parse(item.code_sequence)

                for test_idx, test in enumerate(item.input_tests):
                    self.tracer.reset(indices=test['input'])
                    prog()
                    self.tracer.actions.append(1) # </s>

                    trace_grids.append(self.tracer.grids)
                    conds.append(self.tracer.conds)
                    actions.append([
                        karel_trace.action_to_id[name]
                        for name in self.tracer.actions
                    ])
                    interleave.append([0] + [1, 0] * (len(self.tracer.grids) -
                                                      1))

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
