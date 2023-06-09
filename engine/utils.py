import os
import pickle
from typing import Any

from nltk import tokenize

from engine.preprocess.preprocessor import TextPreprocessor


def preprocess(content: str) -> str:
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(content)

def tokenize_content(content: str) -> list[str]:
    return tokenize.word_tokenize(content)


def save_model(obj: Any, model_name: str):
    dest = os.path.join(models_directory(), model_name)
    with open(dest, 'wb') as f:
        return pickle.dump(obj, f)


def load_model(model_name: str):
    with open(os.path.join(models_directory(), model_name), 'rb') as f:
        return pickle.load(f)
