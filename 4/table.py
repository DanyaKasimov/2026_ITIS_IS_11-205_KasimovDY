import pandas as pd


def build_tf_matrix(tf_csv, output_csv):
    df = pd.read_csv(tf_csv)

    df["doc"] = df["doc"].astype(int)

    matrix = df.pivot(index="term", columns="doc", values="tf").fillna(0)

    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    matrix.to_csv(output_csv, encoding="utf-8")
    print(f"TF matrix saved to {output_csv}")


def build_tfidf_matrix(tfidf_csv, output_csv):
    df = pd.read_csv(tfidf_csv)

    df["doc"] = df["doc"].astype(int)

    matrix = df.pivot(index="term", columns="doc", values="tfidf").fillna(0)

    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    matrix.to_csv(output_csv, encoding="utf-8")
    print(f"TF-IDF matrix saved to {output_csv}")


def build_idf_matrix(idf_csv, tf_csv, output_csv):
    idf_df = pd.read_csv(idf_csv)
    tf_df = pd.read_csv(tf_csv)

    tf_df["doc"] = tf_df["doc"].astype(int)
    docs = sorted(tf_df["doc"].unique())

    idf_matrix = idf_df.set_index("term")

    for doc in docs:
        idf_matrix[doc] = idf_matrix["idf"]

    idf_matrix = idf_matrix[docs]

    idf_matrix.to_csv(output_csv, encoding="utf-8")
    print(f"IDF matrix saved to {output_csv}")


build_tf_matrix("tf.csv", "tf_matrix.csv")
build_idf_matrix("idf.csv", "tf.csv", "idf_matrix.csv")
build_tfidf_matrix("tfidf.csv", "tfidf_matrix.csv")