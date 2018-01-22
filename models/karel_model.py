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
        print("Code:  %s" % ' '.join(item.code_sequence))
        res, = self.inference([item])
        print("Res:   %s" % ' '.join(res.code_sequence))

    def inference(self, batch):
        vocab = self.reset_vocab()
        io_embed = self.encode(vocab, batch)
        init_state = karel.LGRLDecoderState(*self.model.decoder.zero_state(
            io_embed.shape[0] * io_embed.shape[1]))
        memory = karel.LGRLMemory(io_embed)

        sequences = beam_search.beam_search(
            len(batch),
            init_state,
            memory,
            self.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)

        # TODO: deduplicate this with seq2seq_model
        result = [[] for _ in range(len(batch))]
        counters = [0 for _ in range(len(batch))]
        candidates = [[] for _ in range(len(batch))]
        max_eval_trials = self.args.max_eval_trials or self.args.max_beam_trees
        for batch_id, outputs in enumerate(sequences):
            example = batch[batch_id]
            for candidate in outputs[:max_eval_trials]:
                code = [vocab.itos(idx) for idx in candidate]
                counters[batch_id] += 1
                candidates[batch_id].append(code)
                stats = executor.evaluate_code(code, None, example.input_tests,
                        self.executor.execute)
                ok = (stats['correct'] == stats['total'])
                if ok:
                    result[batch_id] = code
                    break
        return [InferenceResult(code_sequence=seq, info={'trees_checked': c, 'candidates': cand})
                for seq, c, cand in zip(result, counters, candidates)]
