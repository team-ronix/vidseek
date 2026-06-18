class EditDistance:
    def __init__(self, word, target):
        self.word = word
        self.target = target
        self.dp = [[-1 for _ in range(len(target) + 1)] for _ in range(len(word) + 1)]
    
    def rec(self, word: str, target: str, i: int, j: int) -> int:
        if (i == len(word)):
            return len(target) - j
        if (j == len(target)):
            return len(word) - i
        if self.dp[i][j] != -1:
            return self.dp[i][j]
        
        if word[i] == target[j]:
            self.dp[i][j] = self.rec(word, target, i + 1, j + 1)
            return self.dp[i][j]

        insert_cost = self.rec(word, target, i, j + 1)
        delete_cost = self.rec(word, target, i + 1, j)
        replace_cost = self.rec(word, target, i + 1, j + 1)

        self.dp[i][j] =  min(insert_cost, delete_cost, replace_cost)
        return self.dp[i][j] + 1
    
    def get_edit_distance(self):
        edit_distance = self.rec(self.word, self.target, 0, 0)
        confidence = 1 - (edit_distance / max(len(self.word), len(self.target)))
        return edit_distance, confidence