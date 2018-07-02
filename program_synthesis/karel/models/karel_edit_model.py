import collections

import numpy as np
import torch
from torch.autograd import Variable

import torchfold

from program_synthesis.common.models import beam_search

from program_synthesis.karel.models import base
from program_synthesis.karel.models import karel_model
from program_synthesis.karel.models import prepare_spec
from program_synthesis.karel.models.modules import karel_common
from program_synthesis.karel.models.modules import karel_edit
from program_synthesis.karel.models.modules import utils
from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.dataset import refine_env
from program_synthesis.karel.dataset import executor
from program_synthesis.karel.dataset import data

# 1. pick among one of the following 9:
#      5 action types (ADD_ACTION)
#      if/repeat/while (WRAP_BLOCK)
#      ifElse (WRAP_IFELSE)
# 2. ADD_ACTION
#      point to insertion location
#    if/ifElse/while
#      pick one condition
#    repeat
#      pick one repeat value
# 3. if/ifElse/while/repeat
#      pick start location
# 4. if/while/repeat
#      pick end location
#    ifElse
#      pick mid location
# 5. ifElse
#      pick end location


class KarelStepEditModel(karel_model.BaseKarelModel):
    def __init__(self, args):
        self.args = args
        self.vocab = data.PlaceholderVocab(
            data.load_vocab(args.word_vocab), args.num_placeholders)
        self.model = karel_edit.KarelEdit(
            len(self.vocab) - args.num_placeholders, args)

        super(KarelStepEditModel, self).__init__(args)

    def debug(self, batch):
        pass

    def eval(self, batch):
        batch_size = batch.batch_size
        if self.args.cuda:
            batch = batch.cuda_infer()

        loss, log_probs_per_example = self.compute_loss_annotated(batch)

        total_log_prob = loss.data[0] * batch_size
        return {'total': batch_size, 'log prob': total_log_prob}

    def compute_loss(self, batch):
        loss, _ = self.compute_loss_annotated(batch)
        return loss

    def compute_loss_annotated(self, batch):
        batch_size = batch.batch_size
        if self.args.cuda:
            batch = batch.cuda_train()

        initial_state, memory, initial_logits = self.model.prepare_initial(
            batch.input_grids, batch.output_grids, batch.current_grids,
            batch.current_code)
        max_code_length = memory.current_code.memory.shape[2]
        initial_state_orig = initial_state
        # state before: (batch x num pairs x hidden,
        #                num layers x batch x num pairs x hidden,
        #                num layers x batch x num pairs x hidden)
        # state after: list of (1 x num pairs x hidden,
        #                       1 x num layers x num pairs x hidden,
        #                       1 x num layers x num pairs x hidden)
        initial_state = zip(
            torch.chunk(initial_state.context, batch_size),
            torch.chunk(initial_state.h.permute(1, 0, 2, 3), batch_size),
            torch.chunk(initial_state.c.permute(1, 0, 2, 3), batch_size))
        # memory: io, current_grid, current_code.memory, current_code.attn_mask
        # before: (batch x num pairs x 512,
        #          batch x num pairs x 256,
        #          batch x num pairs x max code length x 512,
        #          batch x num pairs x max code length)
        # after: list of (1 x num pairs x 512,
        #                 1 x num pairs x 256,
        #                 1 x num pairs x max code length x 512,
        #                 1 x num pairs x max code length)
        memory = zip(*(torch.chunk(t, batch_size) for t in memory.to_flat()))
        initial_logits = torch.chunk(initial_logits, batch_size)

        fold = torchfold.Fold(cuda=self.args.cuda)
        #fold = torchfold.Unfold(nn=self.model, cuda=self.args.cuda)
        zero = fold.add('tf_torch_zero')
        log_probs = []
        for batch_idx, allowed_edits in enumerate(batch.allowed_edits):
            item_log_probs = []
            item_memory = memory[batch_idx]

            # before: 1 x num pairs x length x hidden size
            # after: 1 x length x hidden size
            current_code_memory = item_memory[2][:, 0]
            current_code_attn_mask = item_memory[3][:, 0]

            def step(state, choice_name):
                output, context, h, c = fold.add(
                    'tf_step',
                    fold.add('choice_emb',
                             self.model.choice_vocab.stoi(choice_name)),
                    *(state + item_memory)).split(4)
                return output, (context, h, c)

            def step_pointer(state, loc):
                output, context, h, c = fold.add(
                    'tf_step',
                    fold.add('tf_get_code_emb', current_code_memory, loc),
                    *(state + item_memory)).split(4)
                return output, (context, h, c)

            def log_prob(logits, idx, size):
                assert idx < size
                return fold.add(
                    'tf_get_log_prob:{}'.format(size),
                    fold.add('tf_torch_log_softmax:{}'.format(size), logits),
                    idx)

            def pointer_logits(output, loc):
                assert current_code_attn_mask[0, loc] == 0
                return fold.add('pointer_logits', output, current_code_memory,
                                current_code_attn_mask)

            def batched_sum(v1, v2, v3=zero, v4=zero, v5=zero):
                # v* shape: batch x 1
                return fold.add('tf_batched_sum', v1, v2, v3, v4, v5)

            for action_type, action_args in allowed_edits:
                if action_type == mutation.ADD_ACTION:
                    location, karel_action = action_args

                    action_log_prob = log_prob(
                        initial_logits[batch_idx],
                        self.model.initial_vocab.stoi(karel_action),
                        len(self.model.initial_vocab))

                    output, state = step(initial_state[batch_idx],
                                         karel_action)
                    loc_log_prob = log_prob(
                        pointer_logits(output, location), location, max_code_length)

                    item_log_probs.append(
                        batched_sum(action_log_prob, loc_log_prob))

                elif action_type == mutation.WRAP_BLOCK:
                    block_type, cond_id, start, end = action_args

                    block_type_log_prob = log_prob(
                        initial_logits[batch_idx],
                        self.model.initial_vocab.stoi(block_type),
                        len(self.model.initial_vocab))
                    output, state = step(initial_state[batch_idx], block_type)

                    if block_type == 'repeat':
                        cond_log_prob = log_prob(
                            fold.add('repeat_logits', output), cond_id,
                            len(mutation.REPEAT_COUNTS))
                        cond = len(mutation.CONDS) + cond_id
                    else:
                        cond_log_prob = log_prob(
                            fold.add('cond_logits', output), cond_id,
                            len(mutation.CONDS))
                        cond = cond_id
                    output, state = step(state, cond)

                    start_log_prob = log_prob(
                        pointer_logits(output, start), start, max_code_length)
                    output, state = step_pointer(state, start)

                    end_log_prob = log_prob(
                        pointer_logits(output, end), end, max_code_length)

                    item_log_probs.append(
                        batched_sum(block_type_log_prob, cond_log_prob,
                                    start_log_prob, end_log_prob))

                elif action_type == mutation.WRAP_IFELSE:
                    cond_id, if_start, else_start, end = action_args

                    block_type_log_prob = log_prob(
                        initial_logits[batch_idx],
                        self.model.initial_vocab.stoi('ifElse'),
                        len(self.model.initial_vocab))
                    output, state = step(initial_state[batch_idx], 'ifElse')

                    cond_log_prob = log_prob(
                        fold.add('cond_logits', output), cond_id,
                        len(mutation.CONDS))
                    output, state = step(state, cond_id)

                    if_start_log_prob = log_prob(
                        pointer_logits(output, if_start), if_start, max_code_length)
                    output, state = step_pointer(state, if_start)

                    else_start_log_prob = log_prob(
                        pointer_logits(output, else_start), else_start, max_code_length)
                    output, state = step_pointer(state, else_start)

                    end_log_prob = log_prob(
                        pointer_logits(output, end), end, max_code_length)

                    item_log_probs.append(
                        batched_sum(block_type_log_prob, cond_log_prob,
                                    if_start_log_prob, else_start_log_prob,
                                    end_log_prob))

            if not allowed_edits:
                item_log_probs.append(
                    log_prob(initial_logits[batch_idx],
                             len(self.model.initial_vocab) - 1,
                             len(self.model.initial_vocab)))

            log_probs.append(item_log_probs)

        # log_probs before: list (batch size) of list (allowed_edits)
        # log_probs after: list (batch size) of Tensor, each with length
        #                  `allowed_edits`
        log_probs_t = fold.apply(self.model, log_probs)
        log_probs_per_example = [utils.logsumexp(t) for t in log_probs_t]
        loss = -torch.mean(torch.cat(log_probs_per_example))

        return loss, log_probs_per_example

    def batch_processor(self, for_eval):
        return KarelStepEditBatchProcessor(self.args, self.vocab, for_eval)


class KarelStepEditExample(
        collections.namedtuple('KarelStepEditExample', (
            'batch_size', 'pairs_per_example', 'input_grids', 'output_grids',
            'current_grids', 'current_code', 'allowed_edits',
            'orig_examples'))):
    def cuda_train(self):
        return self.cuda_infer()

    def cuda_infer(self):
        input_grids = self.input_grids.cuda(async=True)
        output_grids = self.output_grids.cuda(async=True)
        current_grids = self.current_grids.cuda(async=True)
        current_code = self.current_code.cuda(async=True)

        return KarelStepEditExample(self.batch_size, self.pairs_per_example,
                                    input_grids, output_grids, current_grids,
                                    current_code, self.allowed_edits,
                                    self.orig_examples)


class KarelStepEditBatchProcessor(object):
    executor = executor.KarelExecutor()

    def __init__(self, args, vocab, for_eval):
        self.args = args
        self.vocab = vocab
        self.for_eval = for_eval

    def __call__(self, batch):
        batch_size = len(batch)
        pairs_per_example = len(batch[0].input_tests)

        input_grids, output_grids = karel_model.encode_io_grids(batch)
        current_grids = torch.zeros(len(batch), 5, 15, 18, 18)

        current_code = prepare_spec.lists_to_packed_sequence(
            [item.cur_code for item in batch],
            self.vocab.stoi,
            cuda=False,
            volatile=False)
        allowed_edits = [item.allowed_edits for item in batch]

        for batch_idx, item in enumerate(batch):
            for test_idx, test in enumerate(item.input_tests):
                result = self.executor.execute(
                    code=item.cur_code,
                    arguments=None,
                    inp=test['input'],
                    record_trace=True)

                # TODO: Try alternatives to result.trace.grids[-1]
                current_grids[batch_idx][test_idx].view(-1)[result.trace.grids[
                    -1]] = 1

        current_grids = Variable(current_grids)

        if self.for_eval:
            orig_examples = batch
        else:
            orig_examples = None

        return KarelStepEditExample(batch_size, pairs_per_example, input_grids,
                                    output_grids, current_grids, current_code,
                                    allowed_edits, orig_examples)
