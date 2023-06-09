# from engine.preprocess.preprocessor import TextPreprocessor
# from engine.preprocess.preprocessor import process_text
from engine.preprocess.preprocessor import TextPreprocessor
from api.http.app import start_server

from engine.tfidf import TfIdfEngine


# test = TfIdfEngine()
# lemmet = LemmatizerWithPOSTagger()
#print(lemmet.lemm())
# def process_text():
#     pass

# def tokenize_text():
#     pass
# print(test.load_vectorizer())

#
# test = TfIdfEngine()
#
# print(test.load_vectorizer())

start_server()