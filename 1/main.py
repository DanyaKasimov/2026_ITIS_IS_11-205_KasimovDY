import requests
import os
import re
import time
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def normalize_url(url):
    return url.split("#")[0].strip()


def download(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = r.apparent_encoding
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            return r.text
    except:
        pass
    return None


def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_russian_words(text):
    words = re.findall(r"\b[а-яА-ЯёЁ]+\b", text)
    return " ".join(words)


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def is_russian(text, threshold=0.8):
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return False
    russian_words = [w for w in words if re.search("[а-яА-ЯёЁ]", w)]
    return len(russian_words) / len(words) >= threshold


def get_links(html, base):
    soup = BeautifulSoup(html, "html.parser")
    result = set()
    for a in soup.find_all("a", href=True):
        link = urljoin(base, a["href"])
        link = normalize_url(link)
        if link.startswith("http"):
            result.add(link)
    return result


def is_valid_link(url):
    url = url.lower()
    return not url.endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".rar"))


def crawl(start_urls, max_pages, min_words, output_dir):
    visited = set()
    queue = deque(start_urls)
    saved = []

    os.makedirs(output_dir, exist_ok=True)
    doc_id = 0

    while queue and len(saved) < max_pages:
        url = normalize_url(queue.popleft())
        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(saved)+1}/{max_pages}] {url}")

        html = download(url)
        if not html:
            continue

        text = clean_text(html)
        wc = word_count(text)

        if wc >= min_words and is_russian(text):
            russian_text = extract_russian_words(text)
            if word_count(russian_text) < min_words:
                print(f"skipped (not enough russian words)")
                continue

            doc_id += 1
            filename = f"doc_{doc_id}.txt"
            path = os.path.join(output_dir, filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(russian_text)

            saved.append((doc_id, url))
            print(f"saved ({len(russian_text.split())} russian words)")

        else:
            print(f"skipped ({wc} total words)")

        for link in get_links(html, url):
            if link not in visited and is_valid_link(link):
                queue.append(link)

        time.sleep(0.5)

    with open(os.path.join(output_dir, "index.txt"), "w", encoding="utf-8") as f:
        for doc_id, url in saved:
            f.write(f"{doc_id} {url}\n")

    print("\nDone")
    print(f"Saved: {len(saved)} pages")
    print(f"Visited: {len(visited)} URLs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+", help="Стартовые URL")
    parser.add_argument("--pages", type=int, default=100, help="Минимальное количество страниц")
    parser.add_argument("--words", type=int, default=1000, help="Минимальное количество слов")
    parser.add_argument("--out", default="pages", help="Директория для сохранения файлов")

    args = parser.parse_args()

    crawl(args.urls, args.pages, args.words, args.out)


if __name__ == "__main__":
    main()