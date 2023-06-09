import re
import string
import os
import pickle
from nltk import pos_tag

from engine.preprocess.date_normalizer import DateNormalizer
from engine.preprocess.spell_check import SpellCheck
from engine.preprocess.abbreviation import Abbreviation

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from engine.preprocess.lemmatizer import LemmatizerWithPOSTagger


class TextPreprocessor:

    def __init__(self) -> None:
        self.stopwords_tokens = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.lemmatizer = LemmatizerWithPOSTagger()
        # self.spell_check = SpellCheck()
        self.abbreviation = Abbreviation()

    def normalize_date(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        return DateNormalizer.normalize_date(tokens)

    def remove_stop_words(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        return " ".join([token for token in tokens if token not in self.stopwords_tokens])

    def remove_urls(self, text: str) -> str:
        return re.sub("http[^\s]*", "", text, flags=re.IGNORECASE)

    def remove_punctuations(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def lower_case_tokens(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        return " ".join([token.lower() for token in tokens])

    def stemmed_tokens(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        return " ".join([self.stemmer.stem(token) for token in tokens])

    def lemmatizing(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        tagged_tokens = pos_tag(tokens)
        return self.lemmatizer.lemmatize(tagged_tokens)

    def spell_checking(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        return self.spell_check.correct(tokens)

    def expand_abbreviation(self, text: str) -> str:
        tokens = self.tokenize_text(text)
        return self.abbreviation.expand_abbreviation(tokens)

    def tokenize_text(self, text: str):
        return text.split(" ")


    def load_vectorizer(self):
        with open(os.path.join("C:\\Users\\iteam\\OneDrive\\Desktop\\IR-project\\ir_models\\tfidf", "vectorizer.pkl"),
                  'rb') as f:
            return pickle.load(f)


    def process_text(self, text: str):
        text = self.tokenize_text(text)
        return " ".join([self.lemmatizing(self.stemmed_tokens(self.remove_stop_words(
            self.remove_punctuations(self.remove_urls(self.lower_case_tokens(word)))))) for word in
                         text])
