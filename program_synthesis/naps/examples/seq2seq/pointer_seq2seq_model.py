import torch
import torch.nn as nn

from program_synthesis.common.models import beam_search

from program_synthesis.naps.examples.seq2seq import decoders, executor, data, pointer_vocab, pointer_prepare
from program_synthesis.naps.examples.seq2seq.base import BaseCodeModel, MaskedMemory, get_attn_mask
from program_synthesis.naps.examples.seq2seq.pointer_seq2seq import PointerSeq2Seq


class PointerSeq2SeqModel(BaseCodeModel):

    def __init__(self, args):
        self.args = args
        self.word_code_vocab = pointer_vocab.load_vocabs(args.word_vocab, args.code_vocab)
        self.model = PointerSeq2Seq(self.word_code_vocab.vocab_size, args)

        self._executor = None
        super(PointerSeq2SeqModel, self).__init__(args)
        self.criterion = nn.NLLLoss(ignore_index=data.PAD_TOKEN)

    @property
    def executor(self):
        if self._executor is None:
            self._executor = executor.UASTExecutor()
        return self._executor

    def encode_(self, masked_padded_texts, text_lengths):
        hidden, memory = self.model.encode(masked_padded_texts, text_lengths)
        attn_mask = get_attn_mask(text_lengths, self.args.cuda)
        return hidden, (memory, attn_mask)

    def decode_(self, masked_padded_codes, padded_texts, hidden, memory, vocab_sizes):
        return self.model.decode(
            masked_padded_codes=masked_padded_codes,
            padded_texts=padded_texts,
            hidden=hidden,
            memory=memory,
            vocab_sizes=vocab_sizes)

    def compute_loss(self, batch, volatile=False):
        (padded_texts, masked_padded_texts, text_lengths, vocabs, vocab_sizes, padded_codes, masked_padded_codes
         ) = pointer_prepare.encode_batch(batch, self.word_code_vocab, self.args.num_placeholders, self.args.cuda, volatile)

        hidden, memory = self.encode_(masked_padded_texts, text_lengths)
        # We use masked code for teacher-forcing, but we use unmasked codes for loss computation.
        logistics = self.decode_(masked_padded_codes, padded_texts, hidden, memory, vocab_sizes)
        # Note, torch.log has issues with computational stability when computing gradients which results in
        # proliferation of NaNs. As per discussion we use clamp: https://github.com/pytorch/pytorch/issues/1620
        logistics = logistics.clamp(min=1e-8)
        labels = padded_codes[:, 1:].contiguous().view(-1)

        return self.criterion(torch.log(logistics), labels)

    def compute_loss_(self, batch, volatile):
        return self.compute_loss(batch, volatile)

    def inference(self, batch):
        (padded_texts, masked_padded_texts, text_lengths, vocabs, vocab_sizes, padded_codes, masked_padded_codes
         ) = pointer_prepare.encode_batch(batch, self.word_code_vocab, self.args.num_placeholders, self.args.cuda,
                                          volatile=True)

        hidden, memory = self.encode_(masked_padded_texts, text_lengths)
        beam_size = self.args.max_beam_trees
        assert beam_size > 1
        def decode_token(prev_tokens, prev_hidden, masked_memory, attentions):
            return self.model.beam_search_decode_token(padded_texts, prev_tokens, prev_hidden, masked_memory,
                                                       vocab_sizes)

        sequences = beam_search.beam_search(
            len(batch),
            decoders.BeamSearchState(
                [hidden for _ in range(self.args.num_decoder_layers)],
                prev_output=None),
            MaskedMemory(memory[0], memory[1]),
            decode_token,
            beam_size,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length)
        return self._try_sequences([vocab.itocode for vocab in vocabs], sequences, batch, beam_size)
