import re
import json
from pathlib import Path
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def add_document(index, documents, doc_name: str, text: str):
    documents.add(doc_name)
    terms = set(tokenize(text))
    for term in terms:
        index[term].add(doc_name)


def build_index_from_folder(folder_path: str):
    folder = Path(folder_path)
    txt_files = sorted(folder.glob("*.txt"))

    index = defaultdict(set)
    documents = set()

    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        add_document(index, documents, file_path.name, text)

    return index, documents


def save_index(index, output_path: str):
    sorted_index = {
        term: sorted(list(docs))
        for term, docs in sorted(index.items(), key=lambda x: x[0])
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_index, f, ensure_ascii=False, indent=4)


def get_docs(index, term: str):
    return index.get(term.lower(), set())


def normalize_query(query: str) -> str:
    query = query.lower()

    query = re.sub(r"\bи\b", "&", query)
    query = re.sub(r"\bили\b", "|", query)
    query = re.sub(r"\bне\b", "!", query)

    query = query.replace("(", " ( ").replace(")", " ) ")
    query = query.replace("&", " & ").replace("|", " | ").replace("!", " ! ")

    query = re.sub(r"\s+", " ", query).strip()
    return query


def tokenize_query(query: str):
    return query.split()


def to_postfix(tokens: list[str]):
    precedence = {"!": 3, "&": 2, "|": 1}
    output = []
    stack = []

    for token in tokens:
        if re.fullmatch(r"[a-zA-Zа-яА-ЯёЁ0-9]+", token):
            output.append(token)
        elif token == "(":
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Ошибка: несогласованные скобки")
            stack.pop()
        elif token in precedence:
            while (
                    stack
                    and stack[-1] in precedence
                    and precedence[stack[-1]] >= precedence[token]
                    and token != "!"
            ):
                output.append(stack.pop())
            stack.append(token)
        else:
            raise ValueError(f"Неизвестный токен: {token}")

    while stack:
        if stack[-1] in ("(", ")"):
            raise ValueError("Ошибка: несогласованные скобки")
        output.append(stack.pop())

    return output


def evaluate_postfix(postfix, index, documents):
    stack = []
    all_docs = set(documents)

    for token in postfix:
        if re.fullmatch(r"[a-zA-Zа-яА-ЯёЁ0-9]+", token):
            stack.append(set(get_docs(index, token)))
        elif token == "!":
            operand = stack.pop()
            stack.append(all_docs - operand)
        elif token == "&":
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right)
        elif token == "|":
            right = stack.pop()
            left = stack.pop()
            stack.append(left | right)

    if len(stack) != 1:
        raise ValueError("Ошибка в выражении")

    return stack[0]


def search(query, index, documents):
    normalized = normalize_query(query)
    tokens = tokenize_query(normalized)
    postfix = to_postfix(tokens)
    result = evaluate_postfix(postfix, index, documents)
    return sorted(result)


def main():
    folder = "../2/processed"
    index_file = "index.json"

    index, documents = build_index_from_folder(folder)
    save_index(index, index_file)

    print(f"Индекс сохранён в {index_file}")
    print(f"Документов: {len(documents)}")
    print(f"Терминов: {len(index)}\n")

    while True:
        query = input("Запрос (exit для выхода): ").strip()
        if query.lower() == "exit":
            break

        try:
            results = search(query, index, documents)
            print("Результат:")
            if results:
                for doc in results:
                    print(" -", doc)
            else:
                print(" Нет совпадений")
        except Exception as e:
            print("Ошибка:", e)

        print()


if __name__ == "__main__":
    main()