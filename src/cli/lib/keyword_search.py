import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from cli.lib.search_utils import CACHE_PATH, load_movies, load_stopwords

stemmer = PorterStemmer()


class invertedIndex:
    def __init__(self):
        self.index = defaultdict(set)  # token: [doc_id1, doc_id2]
        self.docmap = {}  # map document ID: document
        self.term_frequencies = defaultdict(Counter)
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_document(self, term):
        return sorted(self.index[term])

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        return self.term_frequencies[doc_id][token[0]]

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie
        print("Build Success")

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


def tf_command(doc_id, term):
    idx = invertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))


def build_command():
    idx = invertedIndex()
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
    idx = invertedIndex()
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
