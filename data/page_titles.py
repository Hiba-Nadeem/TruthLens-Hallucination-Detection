import pandas as pd
import ast

INPUT_CSV = "../data/fever_1000.csv"
OUTPUT_TXT = "../data/page_titles.txt"

def extract_titles(csv_path):
    df = pd.read_csv(csv_path)
    titles = set()

    for ev in df["evidence"]:
        try:
            ev_list = ast.literal_eval(ev)

            for group in ev_list:
                for item in group:
                    if len(item) >= 3:
                        titles.add(item[2])  # page title

        except:
            continue

    return sorted(titles)


def main():
    titles = extract_titles(INPUT_CSV)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for title in titles:
            f.write(title + "\n")

    print(f"Saved {len(titles)} titles to page_titles.txt")


if __name__ == "__main__":
    main()