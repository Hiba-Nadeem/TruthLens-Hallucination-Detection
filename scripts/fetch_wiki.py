import os
import json
import pandas as pd
import wikipedia
import ast
from tqdm import tqdm

CSV_PATH = "../data/fever_1000.csv"
WIKI_PAGES_DIR = "../data/wiki_pages"
SUMMARIES_PATH = "../data/wiki_summaries.json"


# -----------------------------
# Decode FEVER title
# -----------------------------
def decode_fever_title(title: str) -> str:
    title = title.replace("-LRB-", "(").replace("-RRB-", ")")
    title = title.replace("-LSB-", "[").replace("-RSB-", "]")
    title = title.replace("-LCB-", "{").replace("-RCB-", "}")
    title = title.replace("-COLON-", ":")
    title = title.replace("_", " ")
    return title.strip()


# -----------------------------
# Extract titles from CSV
# -----------------------------
def extract_titles_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    titles = set()

    for ev in df["evidence"]:
        try:
            ev_list = ast.literal_eval(ev)

            for group in ev_list:
                for item in group:
                    # item format: [id1, id2, page_title, sentence_id]
                    if len(item) >= 3:
                        titles.add(item[2])

        except Exception:
            continue

    return list(titles)


# -----------------------------
# Fetch summary
# -----------------------------
def fetch_summary(decoded_title: str):
    try:
        return wikipedia.summary(decoded_title, sentences=5, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
        except:
            return None
    except wikipedia.exceptions.PageError:
        try:
            return wikipedia.summary(decoded_title, sentences=5, auto_suggest=True)
        except:
            return None
    except:
        return None


# -----------------------------
# Load/save helpers
# -----------------------------
def load_existing():
    if os.path.exists(SUMMARIES_PATH):
        with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_summaries(data):
    with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------
# MAIN
# -----------------------------
def main():
    wikipedia.set_lang("en")

    os.makedirs(WIKI_PAGES_DIR, exist_ok=True)

    print("Extracting page titles from CSV...")
    titles = extract_titles_from_csv(CSV_PATH)

    print(f"Total unique titles: {len(titles)}")

    summaries = load_existing()

    if summaries:
        print(f"Resuming: {len(summaries)} already fetched\n")

    failed = []

    for title in tqdm(titles, desc="Fetching summaries"):
        if title in summaries:
            continue

        decoded = decode_fever_title(title)
        text = fetch_summary(decoded)

        if text:
            summaries[title] = text

            # Save individual file
            safe_name = title.replace("/", "_")
            path = os.path.join(WIKI_PAGES_DIR, f"{safe_name}.txt")

            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

        else:
            failed.append(title)

        # Save every 50
        if len(summaries) % 50 == 0:
            save_summaries(summaries)

    save_summaries(summaries)

    print("\nDone!")
    print(f"Fetched: {len(summaries)}")
    print(f"Failed : {len(failed)}")

    if failed:
        print("\nFailed titles:")
        for t in failed[:10]:
            print("-", t)


if __name__ == "__main__":
    main()