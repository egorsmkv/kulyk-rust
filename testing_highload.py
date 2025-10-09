"""
uv run python testing_highload.py
"""

import time
import requests
import polars as pl

addr = "http://127.0.0.1:3000/translate"


def call(text: str, s0: float):
    resp = requests.post(
        addr, json={"text": text, "source_lang": "uk", "target_lang": "en"}
    )
    data = resp.json()
    elapsed = time.time() - s0
    print(elapsed, data)
    return elapsed


if __name__ == "__main__":
    df = pl.read_csv("hf://datasets/speech-uk/text-to-speech-sentences/**/*.csv")

    times = []
    for row in df.iter_rows(named=True):
        s0 = time.time()
        elapsed = call(row["sentence"], s0)
        times.append(elapsed)

    print("Average:", sum(times) / len(times), "sec")
