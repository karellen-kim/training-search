from typing import Union, List

from numpy import ndarray
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
import pandas as pd
import torch

class Searcher :
    def __init__(self):
        # https://huggingface.co/jhgan/ko-sroberta-multitask
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.names = pd.read_csv('./dic.csv', names=['name'], sep='\t')['name'].tolist()
        self.data = self.model.encode(self.names, convert_to_tensor=True)

    def embeddings(self, sentences: List[str]) -> Union[List[Tensor], ndarray, Tensor]:
        return self.model.encode(sentences)

    def search(self, q: str, topK: int = 5):
        # https://github.com/Hong-gi-young/Utube_Contents/blob/af74f3ed6f6a3bf8711a341ec9c3771ef8925138/sentence_transformer.py
        # https://github.com/BM-K/KoSentenceBERT-SKT
        scores = util.pytorch_cos_sim(self.model.encode(q, convert_to_tensor=True), self.data)
        result = torch.topk(scores, k=topK)

        top_result = []
        for i, (score, idx) in enumerate(zip(result[0][0], result[1][0])):
            top_result.append([self.names[idx], '{:.4f}'.format(score)])

        return top_result