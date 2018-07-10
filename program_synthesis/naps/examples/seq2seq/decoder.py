import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F

from program_synthesis.common.modules import attention

class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, num_units, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, num_units))
            input_size = num_units

    def forward(self, inp, hidden):
        hiddens = []
        current = inp
        for i, layer in enumerate(self.layers):
            hidden_i = layer(current, hidden[i])
            current = hidden_i
            if i + 1 != self.num_layers:
                current = self.dropout(current)
            hiddens.append(hidden_i)
        return current, hiddens


class SequenceDecoder(nn.Module):
    def __init__(self, args, input_size, mem_dim, special_tokens_size, vocab_size):
        super(SequenceDecoder, self).__init__()
        self.attention = attention.DotProductAttention(args.num_units, mem_dim, num_heads=args.num_att_heads)
        self.decoder_proj = nn.Linear(args.num_units * 2, args.num_units)
        self.decoder = StackedGRU(args.num_decoder_layers, input_size, args.num_units, args.decoder_dropout)
        self.dropout = nn.Dropout(p=args.decoder_dropout)
        self.output_projection = nn.Linear(args.num_units*args.num_decoder_layers, special_tokens_size + vocab_size,
                                           bias=False)

    def forward(self, enc, hidden, memory):
        memory, attn_mask = memory
        query = hidden[-1]
        context, attn = self.attention(query, memory, attn_mask)
        enc = torch.cat((enc, context), dim=1)
        enc = self.decoder_proj(enc)
        output, hidden = self.decoder(enc, hidden)
        logits = self.output_projection(self.dropout(output))
        return hidden, context, attn, logits


class PointerMechanism(nn.Module):
    def __init__(self, args, special_tokens_size, vocab_size, max_oov_size):
        """
        Augments given decoder inferences with copying.
        """
        super(PointerMechanism, self).__init__()
        self.special_tokens_size = special_tokens_size
        self.vocab_size = vocab_size
        self.max_oov_size = max_oov_size
        self.softmax = nn.Softmax(dim=1)

        self.contexts_w = nn.Linear(args.num_units, 1, bias=False)
        self.hiddens_w = nn.Linear(args.num_decoder_layers*args.num_units, 1, bias=False)
        self.inputs_w = nn.Linear(args.num_units, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def location_logistics(self, attentions, pointer_texts, extended_vocab_size):
        # Sums attentions over tokens in the input padded_texts.
        # Note, in Pytorch the only way to reliably do advanced indexing with repeated indices it by using put_, see the
        # discussion here: https://discuss.pytorch.org/t/indexing-with-repeating-indices-numpy-add-at/10223/11
        # Unfortunately, put_ works with 1D arrays.
        batch_size = attentions.shape[0]
        t_type = (torch.cuda.FloatTensor if attentions.is_cuda else torch.FloatTensor)
        result = Variable(t_type(batch_size * extended_vocab_size).fill_(0), volatile=not self.training)
        # put_ works with flat tensors.
        result = result.put_(pointer_texts.view(-1), attentions.view(-1), accumulate=True)
        return result.view(batch_size, extended_vocab_size)

    def forward(self, decoder_logits, attentions, pointer_texts, contexts, hiddens, inputs):
        attn_logistics = self.location_logistics(attentions, pointer_texts,
                                                 self.special_tokens_size + self.vocab_size + self.max_oov_size)
        # Compute the switch between inferring and copying code tokens.
        # batch x code lengths x 1
        # TODO: Consider using more layers for the switch here.
        switch = self.sigmoid(self.contexts_w(contexts) + self.hiddens_w(hiddens) + self.inputs_w(inputs))
        decoder_logistics = self.softmax(decoder_logits)
        logistics = decoder_logistics * switch
        logistics = F.pad(logistics, (0, self.max_oov_size))
        attn_logistics = attn_logistics * (1.0 - switch)
        return logistics + attn_logistics
