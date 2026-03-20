import os
import re
import argparse
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

import nltk
nltk.download('stopwords')

morph = MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))

def clean_and_lemmatize(text):
    tokens = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)

    lemmas = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in russian_stopwords:
            lemma = morph.parse(token_lower)[0].normal_form
            lemmas.append(lemma)
    return " ".join(lemmas)


def process_documents(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt") and f != "index.txt"]

    for filename in files:
        path_in = os.path.join(input_dir, filename)
        path_out = os.path.join(output_dir, filename)

        with open(path_in, "r", encoding="utf-8") as f:
            text = f.read()

        processed_text = clean_and_lemmatize(text)

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(processed_text)

        print(f"Processed {filename}: {len(processed_text.split())} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Папка с исходными документами")
    parser.add_argument("--output", required=True, help="Папка для сохранения обработанных документов")
    args = parser.parse_args()

    process_documents(args.input, args.output)


if __name__ == "__main__":
    main()