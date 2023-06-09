from typing import List

from nltk.corpus import wordnet


class Abbreviation:
    def __init__(self):
        pass

    def expand_abbreviation(self,words: List[str]):
        return " ".join([wordnet.synsets(word)[0].lemmas()[0].name() for word in words if wordnet.synsets(word)])
