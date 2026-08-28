from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res = []

        for number in numbers:
            lst = []
            number = list(str(number))

            l  = len(number)
            while l > 0 and len(number) > 0:
                temp = "".join(number[0:l])

                if temp in vocab:
                    lst.append(temp)

                    for i in range(l):
                        number.pop(0)

                    l = len(number)
                    continue

                l -= 1

            res.append(lst)

        return res
                

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.

        text = list(text)
        words = []
        word = ""
        for i in range(len(text)):
            if i == len(text)-1:
                word += text[i]
                words.append(word)
                break
            elif text[i] == " ":
                words.append(word)
                word = ""
                words.append(" ")
                continue

            word += text[i]

        print(words)

        count = 0
        for word in words:
            word = list(word)
            l = len(word)
            while l > 0 and len(word) > 0:
                temp = "".join(word[0:l])

                if temp in vocab:
                    count += 1

                    for i in range(l):
                        word.pop(0)

                    l = len(word)
                    continue

                l -= 1

        return count


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        
        token_count = self.count_tokens(text, vocab)
        word_count = len(text.split())

        return round(token_count/word_count, 4)
