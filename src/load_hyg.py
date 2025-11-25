import pandas as pd
from pathlib import Path



#loads the zipped csv into pandas dataframe
#path-/data/hyg/hyg_v38.csv.gz
def load_hyg():
    hyg_path = Path(__file__).resolve().parents[1] / "data" / "hyg" / "hyg_v38.csv.gz"
    
    print(f"Loading from: {hyg_path}")
    df = pd.read_csv(hyg_path)
    print("Loaded dataset with", len(df), "rows")
    return df


#filters dataframe for specific constelattions - just as an example using orion
def filter_constellation(df, const_code):
    const_code = const_code.capitalize()
    subset = df[df["con"] == const_code]
    return subset



#saving each const as its own csv
def save_constellations(df):
    constellations = ["Ori", "Cas", "UMa", "Tau", "Sco", "Cyg"]
    save_folder = Path(__file__).resolve().parents[1] / "data" / "hyg"
    for const in constellations:
        stars = df[df["con"] == const]
        save_path = save_folder / f"{const}_stars.csv"
        stars.to_csv(save_path, index=False)
        print(f"Saved {len(stars)} stars for {const} ? {save_path}")


if __name__ == "__main__":
    df = load_hyg()

    print(sorted(df['con'].dropna().unique()))
    #example orion
    ori = filter_constellation(df, "Ori")
    print(ori.head())

    #saves for all 6
    save_constellations(df)
