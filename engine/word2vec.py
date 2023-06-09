import os

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np


from engine.ir_engine import IREngine
from engine.preprocess.preprocessor import TextPreprocessor
from engine.utils import load_model, load_w2v_model, preprocess, tokenize_content


class Word2VecEngine(IREngine):

    def __init__(self, threshold=0.4) -> None:
        self.model = self._load_model()
        self.matrix = self._load_matrix()
        self.documents_ids = self._load_documents_ids()
        self.threshold = threshold

    @staticmethod
    def _load_model():
        return load_w2v_model(os.path.join("word2vec", "word2vec.model"))

    @staticmethod
    def _load_matrix():
        return load_model(os.path.join("word2vec", "matrix.pk"))

    @staticmethod
    def _load_documents_ids():
        return load_model(os.path.join("word2vec", "documents_ids.pk"))

    def get_documents_count(self) -> int:
        return len(self.documents_ids)

    def _get_embedding_vector(self, doc_tokens):
        embeddings = []
        size = self.model.vector_size
        if len(doc_tokens) < 1:
            return np.zeros(size)
        else:
            for tok in doc_tokens:
                if tok in self.model.wv.index_to_key:
                    embeddings.append(self.model.wv.get_vector(tok))
                else:
                    embeddings.append(np.random.rand(size))
                    
        return np.mean(embeddings, axis=0)


    def preprocess(content) -> str:
        preprocessor = TextPreprocessor()
        return preprocessor.preprocess(content)

    def match_query(self, query: str):
        clean_query = TextPreprocessor.process_text(query)

        query_vector = self._get_embedding_vector(tokenize_content(clean_query))

        matched_documents = []

        for i, document_vector in enumerate(self.matrix):
            similarity = cosine_similarity([document_vector], [query_vector])[0][0]
            if self.threshold is None or similarity >= self.threshold:
                matched_documents.append((self.documents_ids[i], similarity))

        matched_documents.sort(reverse=True, key=lambda d: d[1])

        retrieved_documents = [document for document, _ in matched_documents]

        return retrieved_documents

