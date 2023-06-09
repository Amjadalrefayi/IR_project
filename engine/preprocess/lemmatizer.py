from typing import List

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class LemmatizerWithPOSTagger(WordNetLemmatizer):
    def __init__(self):
        pass

    def _get_wordnet_pos(self, tag: str) -> str:
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(self, words: List[tuple], pos: str = "n") -> str:
        lemmas = []
        for word, word_pos in words:
            lemma = super().lemmatize(word, self._get_wordnet_pos(word_pos))
            lemmas.append(lemma)
        return " ".join(lemmas)