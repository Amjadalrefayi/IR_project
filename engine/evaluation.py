import os
import pickle
import numpy as np


class Evaluation:

    def __init__(self):
        self.qrels = self.load_qrels("")
        self.qrels_dict = self.create_qrels_dict(self.qrels)
        self.queries = self.load_queries_text("")
        self.query_results = self.load_query_results("")
        self.evaluate_result = self.evaluate(self.query_results, self.qrels)


    def evaluation(self):
        return self.evaluate_result
    def precision_at_k(self, retrived_docs_num, relavant_docs_num, k=10):
        if retrived_docs_num == 0 or relavant_docs_num == 0:
            return 0
        if retrived_docs_num <= k:
            return relavant_docs_num / retrived_docs_num
        return relavant_docs_num / k

    def calculate_relevant_count(self, retrieved_docs, query_docs):
        intersect_values = np.intersect1d(retrieved_docs, query_docs)
        matched_count = len(intersect_values)
        return matched_count

    def average_precision(self, retrieved_docs, relevant_docs, k=10):
        total_precisions = 0
        curr_relavant = 1
        if len(relevant_docs) == 0:
            return 0
        if len(retrieved_docs) <= k:
            k = len(retrieved_docs)
        for i in range(k):
            if retrieved_docs[i] in relevant_docs:
                total_precisions += curr_relavant / (i + 1)
                curr_relavant += 1
        matched = self.calculate_relevant_count(retrieved_docs[:k], relevant_docs)
        if matched == 0:
            return 0
        #     print(f"Total Precisions: {total_precisions}")
        #     print(f"Matched: {matched}")
        return total_precisions / matched

    def calculate_map(self, queries, qrles):
        total_ap = 0
        for query_id, docs in queries.items():
            total_ap += self.average_precision(queries.get(query_id)[:10], qrles.get(query_id))
        map_score = total_ap / len(queries)
        return map_score

    def get_first_matched_position(self, qrles, queries):
        for query_id in queries:
            docs_list = queries[query_id]
            queries[query_id] = docs_list[:10]
        matched_positions = {}
        for query_id in queries:
            matched_positions[query_id] = -1
            for i, doc in enumerate(queries[query_id]):
                if doc in qrles[query_id]:
                    matched_positions[query_id] = i
                    break
        return matched_positions

    def reciprocal_rank(self, qrels, queries):
        matched_positions = self.get_first_matched_position(qrels, queries)
        reciprocal_rank = {}
        for query_id, matched_position in matched_positions.items():
            if matched_position == 0:
                reciprocal_rank[query_id] = 0
                continue
            reciprocal_rank[query_id] = 1 / (matched_position + 1)
        return reciprocal_rank

    def mean_reciprocal_rank(self, qrels, queries):
        matched_positions = self.get_first_matched_position(qrels, queries)
        total_rr = 0
        for query_id, matched_position in matched_positions.items():
            if matched_position == -1:
                total_rr += 0
                continue
            #         print(f"Query Id: {query_id} , rr: {1 / (matched_position + 1)}")
            total_rr += (1 / (int(matched_position) + 1))
        mrr = (total_rr) * (1 / len(qrels.items()))
        return mrr

    def recall(self, relavant_docs_num, qrles_docs_num):
        if qrles_docs_num == 0 or relavant_docs_num == 0:
            return 0
        return relavant_docs_num / qrles_docs_num

    def evaluate(self, queries, qrels):
        # Calculate Precision For Every Query
        # Calculate Precision@10 For Every Query
        # Calculate Recall For Every Query
        # Calculate Mean Average Procesion(MAP) For All Queries
        # Calulate Mean Reciprocal Rank (MRR) For All Queries

        Map = self.calculate_map(queries, qrels)
        Mrr = self.mean_reciprocal_rank(qrels, queries)

        result = {
            'queries': []
        }
        for query_id in qrels:
            relevant_docs_num = self.calculate_relevant_count(queries[query_id], qrels[query_id])
            recall_val = self.recall(relevant_docs_num, len(qrels[query_id]))
            avg_precision = self.average_precision(queries[query_id], qrels[query_id])
            precision_at_10 = self.precision_at_k(len(queries[query_id]), relevant_docs_num)

            result['queries'].append({
                'query_id': int(query_id),
                'recall': recall_val,
                'avg_precision': avg_precision,
                'precision@10': precision_at_10,
            })
        result['map'] = Map
        result['mrr'] = Mrr

        return result

    def create_qrels_dict(self, qrels):
        qrels_dict = {}
        for query_id, doc, relevance, _ in qrels:
            if query_id not in qrels_dict:
                qrels_dict[query_id] = []
            if int(relevance) >= 1:
                qrels_dict[query_id].append((doc, int(relevance)))

        # Sort documents for each query ID based on relevance in descending order
        for query_id in qrels_dict:
            qrels_dict[query_id].sort(key=lambda x: x[1], reverse=True)
            qrels_dict[query_id] = [doc for doc, _ in qrels_dict[query_id]]

        return qrels_dict

    def load_query_results(self, file_path):
        with open(file_path, 'rb') as file:
            print(file_path)
            query_documents = pickle.load(file)
        return query_documents

    def load_qrels(self, file_path):
        qrels = []
        with open(file_path, "r") as file:
            for line in file:
                qid, docid, label, iteration = line.strip().split("\t")
                qrels.append((qid, docid, label, iteration))
        return qrels

    def load_queries_text(self, file_path):
        with open(file_path, "r") as file:
            queries = [line.split('\t') for line in file.read().splitlines()]
        return queries