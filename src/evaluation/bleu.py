import sacrebleu

from src.tokenization.tokenizer_utils import decode


def _postprocess(tok, ids):
    text = decode(tok, ids)
    return " ".join(text.split())


def corpus_bleu(hypotheses_ids, reference_ids, or_tokenizer):
    hyps = [_postprocess(or_tokenizer, h) for h in hypotheses_ids]
    refs = [_postprocess(or_tokenizer, r) for r in reference_ids]
    return sacrebleu.corpus_bleu(hyps, [refs])


def corpus_chrf(hypotheses_ids, reference_ids, or_tokenizer):
    """chrF++ (character n-grams + word bigrams, sacrebleu defaults for word_order=2)."""
    hyps = [_postprocess(or_tokenizer, h) for h in hypotheses_ids]
    refs = [_postprocess(or_tokenizer, r) for r in reference_ids]
    return sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs])
