import json
import math
import os

INDEX_PATH = "../3/index.json"
DOCS_DIR = "../2/processed"

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    index = json.load(f)

all_docs = set()
for docs in index.values():
    all_docs.update(docs)

N = len(all_docs)

print(f"Всего документов: {N}")

tf = {}

for term, docs in index.items():
    tf[term] = {}

    for doc in docs:
        doc_path = os.path.join(DOCS_DIR, doc)

        if not os.path.exists(doc_path):
            print(f"Файл не найден: {doc_path}")
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            words = f.read().lower().split()

        total_words = len(words)
        count = words.count(term)

        tf_value = count / total_words if total_words > 0 else 0
        tf[term][doc] = round(tf_value, 6)

idf = {}

for term, docs in index.items():
    df = len(docs)
    idf_value = math.log(N / df) if df > 0 else 0
    idf[term] = round(idf_value, 6)

tf_idf = {}

for term in tf:
    tf_idf[term] = {}

    for doc in tf[term]:
        value = tf[term][doc] * idf[term]
        tf_idf[term][doc] = round(value, 6)

with open("tf.json", "w", encoding="utf-8") as f:
    json.dump(tf, f, indent=4, ensure_ascii=False)

with open("idf.json", "w", encoding="utf-8") as f:
    json.dump(idf, f, indent=4, ensure_ascii=False)

with open("tfidf.json", "w", encoding="utf-8") as f:
    json.dump(tf_idf, f, indent=4, ensure_ascii=False)