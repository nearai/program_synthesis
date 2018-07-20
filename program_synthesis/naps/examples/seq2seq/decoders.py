import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from program_synthesis.common.models import beam_search
from program_synthesis.common.modules import attention

from program_synthesis.naps.examples.seq2seq import data


def argmax_sampler(logits):
    _, ids = logits.data.topk(1)
    return ids[:, :, 0]


class BeamSearchState(beam_search.BeamSearchState):
    __slots__ = ('value', 'prev_output')

    def __init__(self, value, prev_output=None):
        self.value = value
        self.prev_output = prev_output

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        return BeamSearchState([
            v.view(batch_size, -1, v.size(1))[
                indices.data.numpy()] for v in self.value],
                prev_output=self.prev_output.view(batch_size, -1,
                self.prev_output.size(1))[indices.data.numpy()])


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


class BaseSeqDecoderAttn(nn.Module):

    def __init__(self, vocab_size, args, embed=None, input_size=None):
        super(BaseSeqDecoderAttn, self).__init__()
        self.args = args
        self._cuda = args.cuda
        if embed is None:
            self.embed = nn.Embedding(
                vocab_size + args.num_placeholders, args.num_units)
        else:
            self.embed = embed
        input_size = input_size or args.num_units
        self.decoder = StackedGRU(args.num_decoder_layers, input_size, args.num_units, args.decoder_dropout)
        self.out = nn.Linear(
            args.num_units, vocab_size + args.num_placeholders, bias=False)

    def step(self, enc, hidden, memory, prev_output):
        raise NotImplementedError()

    def update_memory(self, memory, hidden):
        return memory

    def forward(self, hidden, memory, outputs, attentions=None, contexts=None, states=None):
        embed = self.embed(data.replace_pad_with_end(outputs))
        preds = []
        hidden = F.dropout(hidden, self.args.decoder_dropout)
        memory = (F.dropout(memory[0], self.args.decoder_dropout), memory[1])
        embed = F.dropout(embed, self.args.decoder_dropout)

        hidden = [hidden for _ in range(self.args.num_decoder_layers)]
        prev_output = None
        for i in range(outputs.size(1)):
            inp = embed[:, i]
            output, hidden, context, attn = self.step(embed[:, i], hidden, memory, prev_output)
            if attentions is not None:
                attentions.append(attn)
            if contexts is not None:
                contexts.append(context)
            if states is not None:
                states.append(hidden)
            memory = self.update_memory(memory, hidden)
            preds.append(output)
            prev_output = output
        preds = torch.stack(preds, dim=1)
        preds = F.dropout(preds, self.args.decoder_dropout)
        return self.out(preds), embed

    def decode_token(self, token, hidden, memory, attentions=None):
        enc = self.embed(token)
        output, hidden, _, attn = self.step(enc, hidden.value, memory, hidden.prev_output)
        if attentions is not None:
            attentions.append(attn)
        # TODO, expand beam search to support changing memory.
        return BeamSearchState(hidden, prev_output=output), self.out(output)

    def sample(self, hidden, memory, sampler=argmax_sampler, attentions=None):
        batch_size = hidden.size(0)
        LongTensor = torch.cuda.LongTensor if self._cuda else torch.LongTensor
        output = [[] for _ in range(batch_size)]
        prev_output = None
        last_input = Variable(LongTensor(
            [0 for _ in range(batch_size)]), volatile=True)
        hidden = BeamSearchState([hidden for _ in range(self.args.num_decoder_layers)])

        for i in range(self.args.max_decoder_length):
            hidden, logits = self.decode_token(
                last_input, hidden, memory, attentions)
            memory = self.update_memory(memory, hidden.value)
            ids = sampler(logits.unsqueeze(1)).squeeze(1)
            last_input = Variable(ids, volatile=True)
            stop = True
            for i in range(batch_size):
                if not output[i] or output[i][-1] != 1:
                    output[i].append(ids[i])
                    stop = False
            if stop:
                break
        return [x[:-1] if x[-1] == 1 else x for x in output]


class SeqDecoderAttn(BaseSeqDecoderAttn):

    def __init__(self, vocab_size, mem_dim, args, embed=None):
        super(SeqDecoderAttn, self).__init__(vocab_size, args, embed=embed)
        self.attention = attention.DotProductAttention(
            args.num_units, mem_dim, num_heads=args.num_att_heads)
        self.decoder_proj = nn.Linear(
            args.num_units * 2, args.num_units)
        
    def step(self, enc, hidden, memory, prev_output):
        memory, attn_mask = memory
        query = hidden[-1]
        context, attn = self.attention(query, memory, attn_mask)
        enc = torch.cat((enc, context), dim=1)
        enc = self.decoder_proj(enc)
        output, hidden = self.decoder(enc, hidden)
        return output, hidden, context, attn
