from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


@dataclass(frozen=True)
class PlayerEvidence:
    name: str
    mafia_probability: float


def _parse_names() -> list[str]:
    print("Enter 8 participant names.")
    print("You can enter comma-separated names or one name per line.")
    first = ("Linas, Paulius, Mazrimas, Lukas, Jasiunas, Dudenas, Emilis, Dominikenas").strip().split(", ")

    if "," in first:
        names = [part.strip() for part in first.split(",") if part.strip()]
    else:
        names = [first] if first else []
        while len(names) < 8:
            value = input(f"Name {len(names) + 1}: ").strip()
            if value:
                names.append(value)

    if len(names) != 8:
        raise ValueError(f"Expected exactly 8 names, got {len(names)}.")

    return names


def _name_features(name: str) -> tuple[float, float, float, float]:
    cleaned = "".join(ch.lower() for ch in name if ch.isalpha())
    if not cleaned:
        cleaned = "a"

    n = len(cleaned)
    vowels = set("aeiou")
    rare = set("qzxjkvw")

    vowel_ratio = sum(ch in vowels for ch in cleaned) / n
    rare_ratio = sum(ch in rare for ch in cleaned) / n
    unique_ratio = len(set(cleaned)) / n

    max_consonants = 0
    run = 0
    for ch in cleaned:
        if ch in vowels:
            run = 0
        else:
            run += 1
            max_consonants = max(max_consonants, run)
    consonant_cluster_ratio = max_consonants / n

    return vowel_ratio, rare_ratio, unique_ratio, consonant_cluster_ratio


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_player_posteriors(names: list[str]) -> list[PlayerEvidence]:
    feature_rows = [_name_features(name) for name in names]
    vowel_values = [row[0] for row in feature_rows]
    rare_values = [row[1] for row in feature_rows]
    unique_values = [row[2] for row in feature_rows]
    cluster_values = [row[3] for row in feature_rows]

    def centered(values: list[float]) -> list[float]:
        mean = sum(values) / len(values)
        return [value - mean for value in values]

    c_vowels = centered(vowel_values)
    c_rare = centered(rare_values)
    c_unique = centered(unique_values)
    c_cluster = centered(cluster_values)

    # Heuristic suspiciousness from name structure, then Bayesian update with prior p(mafia)=2/8.
    # This is a toy model and should be calibrated with real game history for serious use.
    alpha_prior = 1.0
    beta_prior = 3.0
    evidence_strength = 5.0

    results: list[PlayerEvidence] = []
    for i, name in enumerate(names):
        logit = (
            2.4 * c_rare[i]
            + 1.8 * c_cluster[i]
            + 1.2 * c_unique[i]
            - 1.0 * c_vowels[i]
        )
        evidence = _sigmoid(2.0 * logit)
        alpha_post = alpha_prior + evidence_strength * evidence
        beta_post = beta_prior + evidence_strength * (1.0 - evidence)
        mafia_prob = alpha_post / (alpha_post + beta_post)
        results.append(PlayerEvidence(name=name, mafia_probability=mafia_prob))

    return results


def _all_pair_posteriors(players: list[PlayerEvidence]) -> dict[tuple[int, int], float]:
    n = len(players)
    log_scores: dict[tuple[int, int], float] = {}

    for i, j in combinations(range(n), 2):
        log_score = 0.0
        for idx, player in enumerate(players):
            p = min(max(player.mafia_probability, 1e-6), 1.0 - 1e-6)
            log_score += math.log(p if idx in (i, j) else (1.0 - p))
        log_scores[(i, j)] = log_score

    max_log = max(log_scores.values())
    exp_scores = {pair: math.exp(value - max_log) for pair, value in log_scores.items()}
    total = sum(exp_scores.values())
    return {pair: value / total for pair, value in exp_scores.items()}


def _run_bayesian_optimization(pair_posteriors: dict[tuple[int, int], float], seed: int = 7) -> tuple[int, int]:
    ax_client = AxClient(random_seed=seed, verbose_logging=False)
    ax_client.create_experiment(
        name="mafia_pair_optimizer",
        parameters=[
            {
                "name": "mafia_a",
                "type": "range",
                "bounds": [0, 6],
                "value_type": "int",
            },
            {
                "name": "mafia_b",
                "type": "range",
                "bounds": [1, 7],
                "value_type": "int",
            },
        ],
        parameter_constraints=["mafia_a <= mafia_b"],
        objectives={"pair_posterior": ObjectiveProperties(minimize=False)},
        is_test=True,
    )

    def evaluate(params: dict[str, int]) -> float:
        i = int(params["mafia_a"])
        j = int(params["mafia_b"])
        if i == j:
            return 1e-12
        pair = (i, j) if i < j else (j, i)
        return float(pair_posteriors[pair])

    rng = random.Random(seed)
    warmup_pairs = rng.sample(list(pair_posteriors.keys()), k=6)
    for i, j in warmup_pairs:
        _, trial_index = ax_client.attach_trial(parameters={"mafia_a": i, "mafia_b": j})
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"pair_posterior": evaluate({"mafia_a": i, "mafia_b": j})},
        )

    for _ in range(16):
        params, trial_index = ax_client.get_next_trial()
        score = evaluate(params)
        ax_client.complete_trial(trial_index=trial_index, raw_data={"pair_posterior": score})

    best_parameters, _ = ax_client.get_best_parameters()
    a = int(best_parameters["mafia_a"])
    b = int(best_parameters["mafia_b"])
    return (a, b) if a < b else (b, a)


def main() -> None:
    try:
        names = _parse_names()
    except ValueError as error:
        print(f"Input error: {error}")
        return

    players = build_player_posteriors(names)
    pair_posteriors = _all_pair_posteriors(players)
    best_pair = _run_bayesian_optimization(pair_posteriors)

    print("\nEstimated per-player mafia probabilities:")
    for idx, player in enumerate(players, start=1):
        print(f"{idx}. {player.name:<15} p(mafia)={player.mafia_probability:.3f}")

    i, j = best_pair
    top_pair_probability = pair_posteriors[(i, j)]
    print("\nPredicted mafia pair (Bayesian optimization result):")
    print(f"- {players[i].name} and {players[j].name}")
    print(f"- Posterior probability for this exact pair: {top_pair_probability:.3f}")

    top3 = sorted(pair_posteriors.items(), key=lambda item: item[1], reverse=True)[:3]
    print("\nTop 3 pair hypotheses:")
    for rank, ((a, b), prob) in enumerate(top3, start=1):
        print(f"{rank}. {players[a].name} + {players[b].name}: {prob:.3f}")


if __name__ == "__main__":
    main()
