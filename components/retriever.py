import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import string
from tqdm.autonotebook import tqdm
import  utils.variable as variable 
import utils.constant as constant
from pyvi.ViTokenizer import tokenize
class Retriever:
    def __init__(self):
        self.embedding_model_path = variable.embedding_model_path
        self.top_k_retrieval_documents = variable.top_n_retrieval_documents
        self.data_path = variable.full_data_path
        self.embedder = SentenceTransformer(self.embedding_model_path)
        self.full = pd.read_excel(self.data_path)
        self.passages_origin = list(set(self.full['context'].values))
        self.passages=[ tokenize(x) for x in list(set(self.full['context'].values))]
        self.corpus_embeddings = self.embedder.encode(self.passages, convert_to_tensor=True)
        self.tokenized_corpus = self.tokenize_passages(self.passages)
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1)
        self.context_document_name = self.create_context_document_name()
        self.retrieval_weight = constant.retrieval_weight

    def tokenize_passages(self, passages):
        tokenized_corpus = []
        for passage in tqdm(passages):
            tokenized_corpus.append(self.bm25_tokenizer(passage))
        return tokenized_corpus

    def bm25_tokenizer(self, text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)
            if len(token) > 0:
                tokenized_doc.append(token)
        return tokenized_doc

    def create_context_document_name(self):
        context_document_name = dict()
        for index, row in self.full.iterrows():
            context_document_name[row['context']] = row['document']
        return context_document_name

    def retrieval(self, question):
        bm25_scores = self.bm25.get_scores(self.bm25_tokenizer(question))
        documents_score = np.argpartition(bm25_scores, -len(self.passages))[-len(self.passages):]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in documents_score]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['corpus_id'])
        bm25_scores_hits = [item['score'] for item in bm25_hits]

        query_embedding = self.embedder.encode(question, convert_to_tensor=True).to("cuda")
        bi_scores = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=len(self.passages))
        bi_scores = sorted(bi_scores[0], key=lambda x: x['corpus_id'])
        bi_scores = [item['score'] for item in bi_scores]

        min_score = min(bi_scores)
        max_score = max(bi_scores)
        scaled_bi_scores = [(score - min_score) / (max_score - min_score) for score in bi_scores]
        list_score_weight = self.calculate_weighted_scores(bm25_scores_hits, scaled_bi_scores)

        top_n_indices = np.argpartition(-np.array(list_score_weight), self.top_k_retrieval_documents)[:self.top_k_retrieval_documents]
        return self.get_retrieval_results(top_n_indices, list_score_weight)

    def calculate_weighted_scores(self, bm25_scores_hits, scaled_bi_scores):
        list_score_weight = []
        for index in range(len(self.passages)):
            score_bm25 = bm25_scores_hits[index]
            score_bi = scaled_bi_scores[index]
            score_weight = score_bm25 *  self.retrieval_weight + (1 -  self.retrieval_weight) * score_bi
            list_score_weight.append(score_weight)
        return list_score_weight

    def get_retrieval_results(self, top_n_indices, list_score_weight):
        list_retrieval = []
        for score_index in top_n_indices:
            score = list_score_weight[score_index]
            list_retrieval.append({'context': self.passages_origin[score_index], 'score': score})
        return list_retrieval

    def get_context_for_MRC(self, question):
        test_model = pd.DataFrame(columns=['index', 'question', 'document', 'context', 'score', 'new_context'])
        full_retrieval = self.retrieval(question)
        for count, item in enumerate(full_retrieval):
            context = item['context']
            score = item['score']
            document_name = self.context_document_name[context]
            test_model.at[count, 'index'] = 1
            test_model.at[count, 'question'] = question
            test_model.at[count, 'document'] = document_name
            test_model.at[count, 'context'] = context
            test_model.at[count, 'score'] = score
            test_model.at[count, 'new_context'] = question + ".\n" + document_name + " " + context
        return test_model