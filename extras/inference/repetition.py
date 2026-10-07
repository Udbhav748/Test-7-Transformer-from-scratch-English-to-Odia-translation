def banned_ngram_tokens(generated_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return set()

    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[i:i + ngram_size - 1]) == prefix:
            banned.add(generated_ids[i + ngram_size - 1])
    return banned
