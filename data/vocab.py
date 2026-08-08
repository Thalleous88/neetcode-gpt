from typing import Dict, List, Tuple
from collections import Counter

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        text = sorted(set(list(text)))
        stoi = {}
        i = 0
        for t in text:
            stoi[t] = i
            i += 1
        
        itos = {}
        for key, val in stoi.items():
            itos[val] = key
        
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        text = list(text)

        res = [stoi[i] for i in text]

        return res

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        res = [itos[id] for id in ids]
        return "".join(res)
