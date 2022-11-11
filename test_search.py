from unittest import TestCase

from Searcher import Searcher

class TestModel(TestCase):

    def test_embeddings(self):
        search = Searcher()
        sentences = ['안녕하세요?']
        embeddings = search.embeddings(sentences)
        print(embeddings)

    def test_search(self):
        search = Searcher()
        print(search.search('노인을 위한 국가는 없다', 5))
        print(search.search('선샤인', 5))
