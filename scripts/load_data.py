import pandas as pd

def load_fever_data(path):
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_fever_data("../data/fever_1000.csv")

    print("Shape of dataset:", df.shape)
    print("\nColumns:")
    print(df.columns)

    print("\nSample rows:")
    print(df.head(5))