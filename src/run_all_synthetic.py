# src/run_all_synthetic.py

import subprocess
from pathlib import Path

SIMPLE_DIR = Path("data/synthetic/simple")

def extract_result(output: str):

    pred = None
    score = None

    for line in output.splitlines():
        if line.strip().startswith("Predicted Constellation:"):
            pred = line.split(":")[1].strip()
        if line.strip().startswith("Matching Score:"):
            score = line.split(":")[1].strip()

    return pred, score


def main():
    images = sorted(SIMPLE_DIR.glob("*.png"))

    for img in images:
        truth = img.stem 

        proc = subprocess.run(
            ["python", "src/run_pipeline.py", str(img), "--synthetic"],
            capture_output=True,
            text=True
        )

        pred, score = extract_result(proc.stdout)

        if pred is None:
            print(f"{truth:3s} ? ERROR   score=?")
        else:
            print(f"{truth:3s} ? {pred:3s}   score={score}")


if __name__ == "__main__":
    main()
