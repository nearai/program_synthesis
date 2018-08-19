import copy
from itertools import chain

import torch
import torch.nn as nn
from program_synthesis.naps.examples.seq2seq import data, decoders
from program_synthesis.naps.examples.seq2seq.pointer_seq_encoder import SeqEncoder
from torch.autograd import Variable
import torch.nn.functional as F


def location_logistics(attentions, padded_texts, extended_vocab_size):
    # Sums attentions over tokens in the input padded_texts.
    # Note, in Pytorch the only way to reliably do advanced indexing with repeated indices it by using put_, see the
    # discussion here: https://discuss.pytorch.org/t/indexing-with-repeating-indices-numpy-add-at/10223/11
    # Unfortunately, put_ works with 1D arrays.

    # attentions is (batch x code seq len x word seq len). The last dimensions sums to 1.0 and is padded with zeros as
    # the result of exp(-inf).
    # padded_texts is (batch x word seq len) of integers from 0 to vocab_size.
    (batch_size, code_seq_len, word_seq_len) = attentions.shape
    if (batch_size, word_seq_len) != padded_texts.shape:
        # Used in the beam search.
        real_batch_size = padded_texts.shape[0]
        beam_size = batch_size // real_batch_size
        padded_texts = padded_texts.view(real_batch_size, 1, word_seq_len).expand(
            real_batch_size, beam_size, word_seq_len).contiguous().view(-1, word_seq_len)

    padded_texts = data.replace_pad_with_end(padded_texts)
    # Prepare indices for flat attentions.
    # (batch x word_seq_len) - > (batch x code_seq_len x word_seq_len) -> (batch * seq len x seq len)
    indices = padded_texts.unsqueeze(1).expand(-1, code_seq_len, -1).contiguous().view(
        batch_size * code_seq_len, word_seq_len)
    ind_upd = torch.arange(batch_size * code_seq_len).long() * extended_vocab_size
    if indices.is_cuda:
        ind_upd = ind_upd.cuda()
    ind_upd = Variable(ind_upd, volatile=indices.volatile)
    ind_upd = ind_upd.unsqueeze(1).expand(-1, word_seq_len)
    indices = indices + ind_upd

    # Flatten attentions and indices.
    attentions = attentions.view(-1)
    indices = indices.view(-1)

    t_type = (torch.cuda.FloatTensor if attentions.is_cuda else torch.FloatTensor)
    result = Variable(t_type(batch_size * code_seq_len * extended_vocab_size).fill_(0))
    result = result.put_(indices, attentions, accumulate=True)
    return result.view(batch_size, code_seq_len, extended_vocab_size)


class PointerSeq2Seq(nn.Module):

    def __init__(self, word_code_vocab_size, args):
        super(PointerSeq2Seq, self).__init__()
        self.args = args
        self.num_units = args.num_units
        self.bidirectional = args.bidirectional
        self._cuda = args.cuda
        self.word_code_vocab_size = word_code_vocab_size
        self.embed = nn.Embedding(word_code_vocab_size, self.num_units)
        self.encoder = SeqEncoder(args)
        self.softmax = nn.Softmax(dim=2)
        num_directions = 2 if self.bidirectional else 1
        mem_dim = self.num_units * num_directions

        # Decoder decodes into a vocabulary without placeholders.
        decoder_args = copy.deepcopy(args)
        decoder_args.num_placeholders = 0
        self.decoder = decoders.SeqDecoderAttn(
            word_code_vocab_size, mem_dim, decoder_args, embed=self.embed)

        self.contexts_w = nn.Linear(self.args.num_units, 1, bias=False)
        self.states_w = nn.Linear(self.args.num_decoder_layers*self.args.num_units, 1, bias=False)
        self.input_w = nn.Linear(self.args.num_units, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def encode(self, masked_padded_texts, text_lengths):
        masked_padded_texts = self.embed(masked_padded_texts)
        return self.encoder(masked_padded_texts, text_lengths)

    def _joint_logistics(self, decoder_logits, attentions, padded_texts, contexts, states, inputs, vocab_sizes):
        logistics = self.softmax(decoder_logits)
        # batch x code lengths x word_code_placeholders vocab size
        attn_logistics = location_logistics(attentions, padded_texts,
                                            self.word_code_vocab_size + self.args.num_placeholders)

        if not self.training:
            # Mask attn_logistics using vocab_sizes. Note, during the beam search vocab_sizes.shape[0] != batch, but
            # vocab_sizes.shape[0] == batch*num_beams.
            num_beams = attn_logistics.shape[0] // vocab_sizes.shape[0]
            expanded_vocab_sizes = vocab_sizes.unsqueeze(1).expand(
                -1, num_beams).contiguous().view(-1).unsqueeze(1).unsqueeze(2).expand_as(attn_logistics)
            expanded_arange = torch.arange(0, self.word_code_vocab_size + self.args.num_placeholders,
                                           ).long().unsqueeze(0).unsqueeze(1).expand_as(attn_logistics)
            if expanded_vocab_sizes.is_cuda:
                expanded_arange = expanded_arange.cuda()
            # Almost null to avoid division by zero during normalization.
            attn_logistics[expanded_arange >= expanded_vocab_sizes] = 1e-8
            # After masking renormalize logistics. The following discussion suggests to detach the norm variable.
            # https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/3
            attn_logistics_norm = torch.norm(attn_logistics, p=1, dim=2, keepdim=True).detach()
            attn_logistics = attn_logistics.div(attn_logistics_norm.expand_as(attn_logistics))

        # Compute the switch between inferring and copying code tokens.
        # batch x code lengths x 1
        switch = self.sigmoid(self.contexts_w(contexts) + self.states_w(states) + self.input_w(inputs))

        logistics = logistics * switch
        logistics = F.pad(logistics, (0, self.args.num_placeholders))
        attn_logistics = attn_logistics * (1.0 - switch)
        return logistics + attn_logistics

    def decode(self, masked_padded_codes, padded_texts, hidden, memory, vocab_sizes):
        attentions = []  # List of (batch x word seq len) elements.
        contexts = []  # List of (batch x num_units) elements.
        states = []  # List of lists with args.num_decoder_layers elements with (batch x num_units) shape

        # inputs is (batch x code seq len x embedding size).
        logits, inputs = self.decoder(hidden, memory, masked_padded_codes[:, :-1], attentions, contexts, states)

        # batch x code lengths x word lengths. Numbers across the last dim sum to 1.0.
        attentions = torch.stack(attentions, dim=1)
        contexts = torch.stack(contexts, dim=1)
        states = torch.stack(list(chain(*states)), dim=1).view(attentions.shape[0], -1,
                                                               self.args.num_decoder_layers*self.args.num_units)
        joint_logistics = self._joint_logistics(logits, attentions, padded_texts, contexts, states, inputs, vocab_sizes)
        return joint_logistics.view(-1, joint_logistics.size(2))

    def decode_token(self, token, hidden, memory):
        enc = self.decoder.embed(token)
        output, hidden, context, attn = self.decoder.step(enc, hidden.value, memory, hidden.prev_output)
        return decoders.BeamSearchState(hidden, prev_output=output), self.decoder.out(output), context, attn, enc

    def replace_placeholders_with_unk(self, var):
        new_var = var.clone()
        new_var[var >= self.word_code_vocab_size] = data.UNK_TOKEN
        return new_var

    def beam_search_decode_token(self, padded_texts, prev_tokens, prev_hidden, masked_memory, vocab_sizes):
        # Should return:
        # hidden -- BeamSearchState
        # logits - (batch_size*beam_size, vocab_size). Will actually return logistics.
        last_input = self.replace_placeholders_with_unk(prev_tokens)

        hidden, logits, context, attention, input_ = self.decode_token(last_input, prev_hidden, masked_memory)
        states = torch.cat(hidden.value, dim=1).unsqueeze(1)
        joint_logistics = self._joint_logistics(logits.unsqueeze(1),
                                                attention.unsqueeze(1),
                                                padded_texts,
                                                context.unsqueeze(1),
                                                states,
                                                input_.unsqueeze(1),
                                                vocab_sizes)
        joint_logistics = joint_logistics.clamp(min=1e-8)
        return hidden, torch.log(joint_logistics).squeeze(1)
