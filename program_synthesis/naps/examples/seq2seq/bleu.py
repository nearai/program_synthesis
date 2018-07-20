"""Implementation of the BLEU metric copied from
https://github.com/tensorflow/tensor2tensor/blob/758991dd35abc510e93caa5c01c6476bb2380b6e/tensor2tensor/utils/bleu_hook.py
"""
import collections
import math
import numpy as np

from six.moves import xrange
from six.moves import zip


def get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def closest_ref_length(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)
    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
                          (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of references or list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.
    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for (references, translations) in zip(reference_corpus, translation_corpus):
        if not isinstance(references[0], list):
            references = [references]
        translation_length += len(translations)
        reference_length += closest_ref_length(references, len(translations))
        translation_ngram_counts = get_ngrams(translations, max_order)
        
        # Extract union of counts.
        max_ref_counts = {}
        for reference in references:
            ref_ngram_counts = get_ngrams(reference, max_order)
            for ngram in translation_ngram_counts:
                max_ref_counts[ngram] = max(max_ref_counts.get(
                    ngram, 0), ref_ngram_counts[ngram])
        
        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in max_ref_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]
    precisions = [0.0] * max_order
    smooth = 1.0
    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)
