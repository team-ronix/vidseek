confusable = str.maketrans("iIlLJ", "11111")

def norm(c):
    return c.translate(confusable)

def rec(word: str, target: str, dp: list[list[int]],i: int, j: int) -> int:
    if (i == len(word)):
        return len(target) - j
    if (j == len(target)):
        return len(word) - i
    if dp[i][j] != -1:
        return dp[i][j]

    if norm(word[i]) == norm(target[j]):
        dp[i][j] = rec(word, target, dp, i + 1, j + 1)
        return dp[i][j]
    insert_cost = rec(word, target, dp, i, j + 1)
    delete_cost = rec(word, target, dp, i + 1, j)
    replace_cost = rec(word, target, dp, i + 1, j + 1)
    dp[i][j] =  1 + min(insert_cost, delete_cost, replace_cost)
    return dp[i][j]

def get_edit_distance(target, word):
    dp = [[-1 for _ in range(len(target) + 1)] for _ in range(len(word) + 1)]
    edit_distance = rec(word, target, dp, 0, 0)
    confidence = 1 - (edit_distance / max(len(word), len(target)))
    return edit_distance, confidence


def find_closest_word(word_rows, target, threshold=0.6, top_n=5):
    closest_word = -1
    highest_confidence = -1.0

    kept_words = []

    for i, word_row in enumerate(word_rows):
        edit_distance, confidence = get_edit_distance(target.lower(), word_row.word.lower())
        # if confidence > highest_confidence:
        #     highest_confidence = confidence
        #     closest_word = i
        if confidence >= threshold:
            kept_words.append((i, confidence))

    sorted_words = sorted(kept_words, key=lambda x: x[1], reverse=True)[:top_n]
    if (len(sorted_words) == 0):
        return -1
    
    return sorted_words
    