import numpy as np

import torch
from torch.autograd import Variable

from program_synthesis.naps.examples.seq2seq.data import PAD_TOKEN, GO_TOKEN, END_TOKEN
from program_synthesis.naps.examples.seq2seq.pointer_vocab import WordCodePlaceholdersVocab


def texts_to_numpy(texts, word_code_vocab, num_placeholders):
    lengths = [len(t) for t in texts]
    max_length = max(lengths)
    num_texts = len(texts)
    data = np.zeros((num_texts, max_length), dtype=np.int64)
    masked_data = np.zeros((num_texts, max_length), dtype=np.int64)
    vocabs = []
    for i, text in enumerate(texts):
        vocab = WordCodePlaceholdersVocab(word_code_vocab, num_placeholders)
        for j, token in enumerate(text):
            data[i, j], masked_data[i, j] = vocab.wordtoi(token)
        vocabs.append(vocab)
    return data, masked_data, lengths, vocabs


def codes_to_numpy(codes, vocabs):
    max_length = max(len(c) for c in codes) + 2
    num_codes = len(codes)
    # -1: special padding value so that we don't compute the loss over it
    data = np.full((num_codes, max_length), PAD_TOKEN, dtype=np.int64)
    masked_data = np.full((num_codes, max_length), PAD_TOKEN, dtype=np.int64)
    for i, (code, vocab) in enumerate(zip(codes, vocabs)):
        data[i, 0], masked_data[i, 0] = GO_TOKEN, GO_TOKEN  # Start with <S>
        for j, token in enumerate(code):
            data[i, j + 1], masked_data[i, j + 1] = vocab.codetoi(token)
        data[i, len(code) + 1], masked_data[i, len(code) + 1] = END_TOKEN, END_TOKEN  # End with </S>
    return data, masked_data


def texts_to_vars(texts, word_code_vocab, num_placeholders, cuda, volatile):
    np_texts, np_masked, lengths, vocabs = texts_to_numpy(texts, word_code_vocab, num_placeholders)
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    v_texts = Variable(tensor_type(np_texts), volatile=volatile)
    v_masked = Variable(tensor_type(np_masked), volatile=volatile)
    return v_texts, v_masked, lengths, vocabs


def codes_to_vars(codes, vocabs, cuda, volatile):
    np_codes, np_masked = codes_to_numpy(codes, vocabs)
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    v_codes = Variable(tensor_type(np_codes), volatile=volatile)
    v_masked = Variable(tensor_type(np_masked), volatile=volatile)
    return v_codes, v_masked


def prepare_code_sequence(examples):
    codes = []
    for example in examples:
        codes.append(example["code_sequence"])
    return codes


def get_vocab_size(vocabs, cuda, volatile):
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    return tensor_type([v.vocab_size for v in vocabs])


def encode_batch(batch, word_code_vocab, num_placeholders, cuda, volatile):
    texts = [ex["text"] for ex in batch]
    padded_texts, masked_padded_texts, text_lengths, vocabs = texts_to_vars(
        texts, word_code_vocab, num_placeholders, cuda, volatile)

    codes = prepare_code_sequence(batch)
    padded_codes, masked_padded_codes = codes_to_vars(
        codes, vocabs, cuda, volatile)

    vocab_sizes = get_vocab_size(vocabs, cuda, volatile)

    return padded_texts, masked_padded_texts, text_lengths, vocabs, vocab_sizes, padded_codes, masked_padded_codes
