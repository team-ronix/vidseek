import string

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "did", "do", "does", "doing", "for", "from", "had", "has", "have",
    "having", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "no", "not",
    "now", "of", "on", "once", "only", "or", "other", "our", "out", "own",
    "she", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "up",
    "us", "very", "was", "we", "were", "what", "when", "where", "which",
    "who", "will", "with", "you", "your",
})

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    tokens = text.lower().translate(_PUNCT_TABLE).split()
    tokens = [t for t in tokens if len(t) > 1]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens
