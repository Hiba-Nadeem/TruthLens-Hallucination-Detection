import pandas as pd
import ast
from tqdm import tqdm


def extract_page_titles(evidence_str):
    """
    Extract page titles from FEVER evidence field
    """
    pages = set()

    try:
        evidence = ast.literal_eval(evidence_str)

        for group in evidence:  # multiple evidence groups
            for item in group:  # each evidence item
                page_title = item[2]

                if page_title is not None:
                    pages.add(page_title)

    except:
        pass  # skip malformed rows

    return pages


def main():
    df = pd.read_csv("../data/fever_1000.csv")

    all_pages = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pages = extract_page_titles(row["evidence"])
        all_pages.update(pages)

    print(f"\nTotal unique pages: {len(all_pages)}")

    # Save pages
    with open("../data/page_titles.txt", "w", encoding="utf-8") as f:
        for page in sorted(all_pages):
            f.write(page + "\n")

    print("Page titles saved to data/page_titles.txt")


if __name__ == "__main__":
    main()