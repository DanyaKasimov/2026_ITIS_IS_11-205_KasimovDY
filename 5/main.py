import os
import re
import math
import pandas as pd
from collections import Counter


PROCESSED_DIR = "../2/processed"
TFIDF_PATH = "../4/tfidf.csv"
IDF_PATH = "../4/idf.csv"
RESULTS_PATH = "vector_search_results.csv"


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\sа-яё]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_query(query):
    query = normalize_text(query)
    return query.split() if query else []


def detect_column(df, possible_names):
    lower_map = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_idf_from_csv(tfidf_csv_path, idf_csv_path=None):
    idf_dict = {}

    if os.path.exists(tfidf_csv_path):
        df = pd.read_csv(tfidf_csv_path)

        term_col = detect_column(df, ["term"])
        idf_col = detect_column(df, ["idf"])

        if term_col and idf_col:
            tmp = df[[term_col, idf_col]].dropna().drop_duplicates()
            for _, row in tmp.iterrows():
                term = str(row[term_col]).strip()
                try:
                    idf = float(row[idf_col])
                    idf_dict[term] = idf
                except ValueError:
                    pass

    if not idf_dict and idf_csv_path and os.path.exists(idf_csv_path):
        df = pd.read_csv(idf_csv_path)

        term_col = detect_column(df, ["term"])
        idf_col = detect_column(df, ["idf"])

        if term_col and idf_col:
            tmp = df[[term_col, idf_col]].dropna().drop_duplicates()
            for _, row in tmp.iterrows():
                term = str(row[term_col]).strip()
                try:
                    idf = float(row[idf_col])
                    idf_dict[term] = idf
                except ValueError:
                    pass

    return idf_dict


def read_processed_documents(processed_docs_dir):
    documents = {}

    for filename in os.listdir(processed_docs_dir):
        doc_id = os.path.splitext(filename)[0]
        path = os.path.join(processed_docs_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        tokens = text.split()
        if tokens:
            documents[doc_id] = tokens

    return documents


def build_document_vectors(documents, idf_dict):
    doc_vectors = {}
    doc_norms = {}

    for doc_id, tokens in documents.items():
        counts = Counter(tokens)
        total_terms = len(tokens)

        vector = {}
        for term, count in counts.items():
            if term not in idf_dict:
                continue
            tf = count / total_terms
            tfidf = tf * idf_dict[term]
            vector[term] = tfidf

        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        doc_vectors[doc_id] = vector
        doc_norms[doc_id] = norm

    return doc_vectors, doc_norms


def build_query_vector(query_terms, idf_dict):
    counts = Counter(query_terms)
    total_terms = len(query_terms)

    vector = {}
    for term, count in counts.items():
        if term not in idf_dict:
            continue
        tf = count / total_terms
        tfidf = tf * idf_dict[term]
        vector[term] = tfidf

    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    return vector, norm


def cosine_similarity(vec1, norm1, vec2, norm2):
    if norm1 == 0 or norm2 == 0:
        return 0.0

    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

    return dot_product / (norm1 * norm2)


def search(query, idf_dict, doc_vectors, doc_norms, top_k=None):
    query_terms = parse_query(query)

    if not query_terms:
        return []

    query_vector, query_norm = build_query_vector(query_terms, idf_dict)

    if not query_vector:
        return []

    results = []
    for doc_id, doc_vector in doc_vectors.items():
        sim = cosine_similarity(query_vector, query_norm, doc_vector, doc_norms[doc_id])
        if sim > 0:
            results.append((doc_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def save_results_for_google_sheets(all_results, output_csv_path, all_documents):
    def extract_number(doc_id):
        doc_id = str(doc_id)
        return int(''.join(filter(str.isdigit, doc_id))) if any(c.isdigit() for c in doc_id) else 0

    all_doc_ids = list(all_documents.keys())

    query_maps = {}
    for query, results in all_results.items():
        query_maps[query] = {
            extract_number(doc_id): round(sim, 6)
            for doc_id, sim in results
        }

    def sort_key(doc_id):
        doc_num = extract_number(doc_id)

        if all_results:
            first_query = list(all_results.keys())[0]
            sim = query_maps[first_query].get(doc_num, 0.0)
        else:
            sim = 0.0

        return (doc_num, -sim)

    sorted_doc_nums = sorted(
        [extract_number(doc_id) for doc_id in all_doc_ids],
        key=sort_key
    )

    data = {"doc": sorted_doc_nums}

    for query in all_results:
        sim_map = query_maps[query]
        data[query] = [sim_map.get(doc_num, 0.0) for doc_num in sorted_doc_nums]

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def print_results(query, results):
    print(f"\nЗапрос: {query}")

    print(f"{'Документ':<15} {'Сходство':<12}")
    print("-" * 30)
    for doc_id, score in results:
        print(f"{doc_id:<15} {score:.6f}")



def main():
    idf_dict = load_idf_from_csv(TFIDF_PATH, IDF_PATH)
    documents = read_processed_documents(PROCESSED_DIR)
    doc_vectors, doc_norms = build_document_vectors(documents, idf_dict)

    all_results = {}

    while True:
        query = input("Введите запрос: ").strip()
        if not query:
            break

        results = search(query, idf_dict, doc_vectors, doc_norms)
        print_results(query, results)
        all_results[query] = results

    save_results_for_google_sheets(all_results, RESULTS_PATH, documents)
    print(f"\nРезультаты сохранены в файл: {RESULTS_PATH}")


if __name__ == "__main__":
    main()