import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from cli.lib.search_utils import BM25_B, BM25_K1, CACHE_DIR, load_movies, load_stopwords

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)  # token: [doc_id1, doc_id2]
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap = {}  # map document ID: document
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = CACHE_DIR / "term_frequencies.pkl"
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_average_doc_lengths(self):
        lengths = list(self.doc_lengths.values())
        total = 0
        if len(lengths) == 0:
            return 0.0
        for l in lengths:
            total += l
        return total / len(lengths)

    def get_document(self, term):
        return sorted(self.index[term])

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        return self.term_frequencies[doc_id][token[0]]

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_average_doc_lengths()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        token = token[0]
        doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        token = token[0]
        doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        # log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log(
            (doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1
        )

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie
        print("Build Success")

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, term)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # docs = idx.get_document("merida")
    # print(f"First document for token 'merida' = {docs[0]}")


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text):
    text = clean_text(text)
    stopwodrs = load_stopwords()
    res = []

    def _filters(tok):
        tok = tok.strip("\n")
        if tok and tok not in stopwodrs:
            return True

    for tok in text.split():
        if _filters(tok):
            tok = stemmer.stem(tok)
            res.append(tok)
    return res


def search_command(query, n_result=5):
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    # iterate over each token in the query
    for qt in query_tokens:
        # use the inverted index to get matching documents.
        matching_doc_ids = idx.get_document(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_document = idx.docmap[matching_doc_id]
            res.append(matching_document)
            # Stop searching once you have 5 or n_result.
            if len(res) >= n_result:
                return res
    return res
