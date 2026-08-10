import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        positive = [p.split() for p in positive]
        negative = [n.split() for n in negative]

        lookup = {}
        vocab = sorted(set([word for sentence in positive + negative for word in sentence]))
        
        for i, word in enumerate(vocab):
            lookup[word] = float(i+1)
        

        positive = [torch.tensor([lookup[i] for i in sentence]) for sentence in positive]
        negative = [torch.tensor([lookup[i] for i in sentence]) for sentence in negative]

        res = positive + negative

        res = nn.utils.rnn.pad_sequence(res, padding_value=0, batch_first=True)

        return res
      

        

