import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from engine.preprocess.preprocessor import TextPreprocessor


class TfIdfEngine:

    def __init__(self) -> None:
        self.vectorizer = self.load_vectorizer()
        self.matrix = self.load_matrix()
        self.documents_ids = self.load_documents_matrix()
        self.textProcessor = TextPreprocessor()
        self.threshold = 0.4

    @staticmethod
    def load_vectorizer():
        print("TESTTTTTTTTTTTTTTTTTTTTTTTT")
        with open(os.path.join("/Users/akhateeb22/Desktop/IR-project/ir_models/tfidf", "vectorizer.pkl"),
                  'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_matrix():
        with open(os.path.join("/Users/akhateeb22/Desktop/IR-project/ir_models/tfidf", "tfidf_matrix.pkl"),
                  'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_documents_matrix():
        with open(os.path.join("/Users/akhateeb22/Desktop/IR-project/ir_models/tfidf", "document_names.pkl"),
                  'rb') as f:
            return pickle.load(f)

    def get_documents_count(self) -> int:
        return len(self.documents_ids)

    def match_query(self, query: str) -> list[str]:
        clean_query = self.textProcessor.process_text(query)
        query_tfidf = self.vectorizer.transform([clean_query])
        query_vector = query_tfidf.toarray()[0]

        matched_documents = []
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.matrix)
        sorted_indices = similarities.argsort()[0][::-1]
        k = 20
        docs = []
        print(f"Top {k} most similar documents:")
        count = 0
        for i in range(len(sorted_indices)):
            doc_index = sorted_indices[i]
            similarity = similarities[0][doc_index]

            # Exclude documents with similarity of 0
            if similarity == 0:
                continue

            doc_name = self.documents_ids[doc_index]
            print(f"Document: {doc_name}, Similarity: {similarity}")
            docs.append(doc_name)
            count += 1

            # Break the loop if k documents have been printed
            if count == k:
                break

        return docs

    def get_file_content(self, file_name):
        with open(f"/Users/akhateeb22/Desktop/antique_documents/{file_name}.txt", 'r') as file:
            content = file.read()
        return content

