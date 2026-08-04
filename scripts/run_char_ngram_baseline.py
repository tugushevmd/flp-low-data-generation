from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import audit_generation


DATA_DIR = ROOT / "data" / "flp_curation"
CURATION_DIR = DATA_DIR
RESULT_DIR = ROOT / "results" / "baselines" / "char_5gram"

N = 5
GENERATION_SEEDS = [101, 202, 303]
SAMPLES_PER_SEED = 1000


def read_smiles(path, column="smiles"):
    return pd.read_csv(path)[column].dropna().astype(str).tolist()


def fit_ngram(smiles, n=N):
    counts = {
        order: defaultdict(Counter)
        for order in range(n)
    }

    for value in smiles:
        text = "^" * (n - 1) + value + "$"
        for position in range(n - 1, len(text)):
            history = text[:position]
            next_character = text[position]
            for order in range(n):
                counts[order][history[-order:] if order else ""][next_character] += 1

    return counts


def next_character(counts, history, rng):
    for order in range(N - 1, -1, -1):
        context = history[-order:] if order else ""
        options = counts[order].get(context)
        if options:
            characters = list(options)
            probabilities = np.array(list(options.values()), dtype=float)
            probabilities /= probabilities.sum()
            return rng.choice(characters, p=probabilities)

    return "$"


def generate_one(counts, rng, max_chars):
    history = "^" * (N - 1)
    result = []

    for _ in range(max_chars):
        character = next_character(counts, history, rng)
        if character == "$":
            return "".join(result), False
        result.append(character)
        history += character

    return "".join(result), True


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_smiles = read_smiles(CURATION_DIR / "train_smiles.csv")
    reference_smiles = read_smiles(
        CURATION_DIR / "reference_smiles_curated_v2.csv",
        "canonical_smiles",
    )
    template_smiles = read_smiles(
        DATA_DIR / "template_candidates.csv",
        "canonical_smiles",
    )

    counts = fit_ngram(train_smiles)
    max_chars = max(map(len, train_smiles)) + 20
    summaries = []

    for seed in GENERATION_SEEDS:
        rng = np.random.default_rng(seed)
        generated = [
            generate_one(counts, rng, max_chars)
            for _ in range(SAMPLES_PER_SEED)
        ]
        raw_smiles = [item[0] for item in generated]
        reached_limit = [item[1] for item in generated]

        rows, summary, funnel = audit_generation(
            raw_smiles=raw_smiles,
            train_smiles=train_smiles,
            seed_smiles=reference_smiles,
            template_smiles=template_smiles,
            reached_max_length=reached_limit,
        )
        summary.insert(0, "generation_seed", seed)
        summaries.append(summary)

        pd.DataFrame({
            "generated_smiles": raw_smiles,
            "reached_max_length": reached_limit,
        }).to_csv(RESULT_DIR / f"samples_seed_{seed}.csv", index=False)
        rows.to_csv(RESULT_DIR / f"evaluated_seed_{seed}.csv", index=False)
        funnel.to_csv(RESULT_DIR / f"funnel_seed_{seed}.csv", index=False)

        print(
            f"seed {seed}: validity {summary.at[0, 'validity']:.3f}, "
            f"strict FLP {summary.at[0, 'strict_flp_yield']:.3f}, "
            f"final candidates {summary.at[0, 'final_candidate_yield']:.3f}"
        )

    run_summary = pd.concat(summaries, ignore_index=True)
    run_summary.to_csv(RESULT_DIR / "run_summary.csv", index=False)

    metric_columns = [
        column
        for column in run_summary.columns
        if column not in {"generation_seed", "evaluator_version"}
    ]
    aggregate = pd.DataFrame({
        "metric": metric_columns,
        "mean": [run_summary[column].mean() for column in metric_columns],
        "sd": [run_summary[column].std() for column in metric_columns],
    })
    aggregate.to_csv(RESULT_DIR / "aggregate_summary.csv", index=False)

    plot_metrics = [
        ("validity", "Валидность"),
        ("unique_valid_yield", "Уникальные валидные"),
        ("strict_flp_yield", "Строгие FLP-like"),
        ("final_candidate_yield", "Финальные кандидаты"),
    ]
    means = [run_summary[column].mean() for column, _ in plot_metrics]
    errors = [run_summary[column].std() for column, _ in plot_metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [label for _, label in plot_metrics],
        means,
        yerr=errors,
        color=["#356859", "#7B6D8D", "#8C3B4A", "#6B7280"],
        capsize=4,
    )
    ax.set_ylabel("Доля от всех генераций")
    ax.set_title("Символьная 5-граммная модель")
    ax.set_ylim(0, max(0.1, max(means) + max(errors) + 0.02))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "char_5gram_summary.png", dpi=180)

    print()
    print(aggregate.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
