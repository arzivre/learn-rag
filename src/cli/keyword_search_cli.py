import argparse

from cli.lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from cli.lib.search_utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build cache")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Calculate term frequencies")
    tf_parser.add_argument("doc_id", type=int, help="document ID to check")
    tf_parser.add_argument("term", type=str, help="Term to tokenize and get tf for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate inverse document frequencies"
    )
    idf_parser.add_argument("term", type=str, help="Term to find idf score for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate term frequencies - inverse document frequencies"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="document ID to check")
    tfidf_parser.add_argument("term", type=str, help="Search term to find counts for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i}. {result['title']}")

        case "build":
            build_command()

        case "tf":
            tf_command(args.doc_id, args.term)

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
