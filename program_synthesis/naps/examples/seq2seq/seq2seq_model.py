import torch
import torch.nn as nn


from . import encoder
from . import decoder
from . import batch_preparator


class SequentialText2Uast(nn.Module):
    def __init__(self, args, vocab):
        super(SequentialText2Uast, self).__init__()
        self.args = args
        self.num_units = args.num_units
        self.bidirectional = args.bidirectional
        self._cuda = args.cuda
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_oov_size = args.max_oov_size
        self.max_queue_size = 1

        self.encoder = encoder.SeqEncoder(args)
        num_directions = 2 if self.bidirectional else 1
        mem_dim = self.num_units * num_directions
        self.decoder = decoder.SequenceDecoder(args, args.num_units, mem_dim, batch_preparator.SPECIAL_SIZE,
                                               self.vocab_size)
        self.pointer_mechanism = decoder.PointerMechanism(args, batch_preparator.SPECIAL_SIZE, self.vocab_size,
                                                          args.max_oov_size)
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(p=args.decoder_dropout)
        self.criterion = nn.NLLLoss(ignore_index=-1)
        if self._cuda:
            self.cuda()
        self.preparator = batch_preparator.BatchPreparator(args, self.vocab)

    def run_encoder(self, encoder_input, texts_lengths):
        encoder_input = self.dropout(encoder_input)
        hidden, memory = self.encoder(encoder_input, texts_lengths)
        hidden = [self.dropout(hidden) for _ in range(self.args.num_decoder_layers)]
        memory = self.dropout(memory)
        return hidden, memory

    def compute_loss(self, batch):
        pointer_texts, texts_lengths, encoder_input, attn_mask, oovs = self.preparator.process_texts(batch)
        programs_data, forced_output, max_program_len = self.preparator.process_golden_programs(batch, oovs)
        start_tokens = self.preparator.get_start_tokens(len(batch))
        if self._cuda:
            pointer_texts = pointer_texts.cuda()
            encoder_input = encoder_input.cuda()
            attn_mask = attn_mask.cuda()
            programs_data = programs_data.cuda()
            forced_output = forced_output.cuda()
            start_tokens = start_tokens.cuda()
        hidden, memory = self.run_encoder(encoder_input, texts_lengths)
        all_logistics = []
        tokens_to_supervise = start_tokens
        for step in range(max_program_len):
            hidden, context, attn, decoder_logits = self.decoder(tokens_to_supervise, hidden, (memory, attn_mask))
            joint_logistics = self.pointer_mechanism(decoder_logits,
                                                     attn,
                                                     pointer_texts,
                                                     context,
                                                     torch.cat(hidden, dim=1),
                                                     tokens_to_supervise)
            all_logistics.append(joint_logistics)
            tokens_to_supervise = forced_output[:, step, :]
        all_logistics = torch.cat(all_logistics, dim=0)
        all_logistics = all_logistics.clamp(min=1e-10)  # Avoid issues with log.
        all_logits = torch.log(all_logistics)
        return self.criterion(all_logits, programs_data.view(-1))
