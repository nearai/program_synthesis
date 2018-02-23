import collections

import numpy as np
import torch
from torch.autograd import Variable

from program_synthesis.datasets import executor
from program_synthesis.datasets.karel import karel_runtime
from program_synthesis.models import (
    base,
    beam_search,
    karel_model,
    prepare_spec,
)
from program_synthesis.models.modules import karel


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
