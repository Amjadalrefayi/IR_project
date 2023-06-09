import os
import pickle
import enchant
import nltk
from spellchecker import SpellChecker
from nltk import tokenize

from sklearn.metrics.pairwise import cosine_similarity
from engine.preprocess.preprocessor import TextPreprocessor


class Prediction:

    def __init__(self) -> None:
        self.vectorizer = self.load_vectorizer()
        self.matrix = self.load_matrix()
        self.queries = self.load_queries()

    @staticmethod
    def load_vectorizer():
        with open(os.path.join("/Users/akhateeb22/Desktop/IR-project/ir_models/tfidf", "vectorizer.pkl"),
                  'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_matrix():
        with open(os.path.join("/Users/akhateeb22/Desktop/IR-project/ir_models/tfidf", "tfidf_matrix.pkl"),
                  'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_queries():
        with open(os.path.join("/Users/akhateeb22/Desktop/IR-project/ir_models/prediction", "queries.pkl"),
                  'rb') as f:
            return pickle.load(f)

    def tokenize(self, text: str):
        tokens = nltk.word_tokenize(text)
        return tokens


    def correct_query(self, query):
        tokens = self.tokenize(query)
        spell = SpellChecker()
        misspelled = spell.unknown(tokens)
        for i, token in enumerate(tokens):
            if token in misspelled:
                corrected = spell.correction(token)
                if (corrected != None):
                    tokens[i] = corrected

        corrected_query = ' '.join(tokens)

        return corrected_query

