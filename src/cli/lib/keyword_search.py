import string

from nltk.stem import PorterStemmer

from cli.lib.search_utils import load_movies, load_stopwords

stemmer = PorterStemmer()


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


def has_matching_token(query_tokens, movie_tokens):
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False


def search_command(query, n_result):
    movies = load_movies()
    res = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, movie_tokens):
            res.append(movie)
        if len(res) == n_result:
            break
    return res
