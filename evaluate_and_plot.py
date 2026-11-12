"""
Rôle C — Évaluation, courbes d'apprentissage (avec suivi EvalCallback) et analyse comparative
Module : Apprentissage par renforcement — Projet ISEN Nantes (M2 DP IA)
"""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean_std(values: List[float], window: int = 50):
    """Calcule la moyenne glissante et l'écart-type glissant."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr, arr
    if len(arr) < window:
        window = max(1, len(arr) // 4)
    means = np.empty_like(arr)
    stds = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        segment = arr[start : i + 1]
        means[i] = np.mean(segment)
        stds[i] = np.std(segment)
    return means, stds


def generate_plots(log_dir: str = "logs", out_dir: str = "plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Chargement de la baseline hasard
    random_path = os.path.join(log_dir, "random_baseline.json")
    random_reward = -7.3
    if os.path.exists(random_path):
        with open(random_path, "r", encoding="utf-8") as f:
            random_data = json.load(f)
            random_reward = random_data.get("mean_raw_reward", -7.3)

    runs_config = [
        (
            "flappy_ppo_shaped_metrics.json",
            "PPO + Reward Shaping + EvalCallback (Retenu)",
            "#1f77b4",
        ),
        ("flappy_ppo_default_metrics.json", "PPO + Récompense par défaut", "#ff7f0e"),
        ("flappy_dqn_shaped_metrics.json", "DQN + Reward Shaping", "#2ca02c"),
    ]

    loaded_runs: List[Dict] = []
    for fname, label, color in runs_config:
        fpath = os.path.join(log_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["label"] = label
                data["color"] = color
                loaded_runs.append(data)

    if not loaded_runs:
        print("Aucun fichier de log trouvé dans 'logs/'. Lancez d'abord train.py.")
        return

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # -------------------------------------------------------------------------
    # Figure 1 : Courbe d'apprentissage principale (PPO + EvalCallback vs Hasard)
    # -------------------------------------------------------------------------
    main_run = loaded_runs[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    ts = np.array(main_run["timesteps"])
    raw_r = main_run["raw_rewards"]
    scores = main_run["scores"]

    r_mean, _ = rolling_mean_std(raw_r, window=50)
    s_mean, _ = rolling_mean_std(scores, window=50)

    ax1.plot(
        ts,
        raw_r,
        color=main_run["color"],
        alpha=0.18,
        linewidth=0.8,
        label="Récompense par épisode (brute)",
    )
    ax1.plot(
        ts,
        r_mean,
        color=main_run["color"],
        linewidth=2.4,
        label="Moyenne glissante (entraînement)",
    )

    eval_hist = main_run.get("eval_history", [])
    best_ts = main_run.get("best_checkpoint_timestep", None)
    if eval_hist:
        ev_ts = [e["timestep"] for e in eval_hist]
        ev_r = [e["mean_raw_reward"] for e in eval_hist]
        ax1.plot(
            ev_ts,
            ev_r,
            color="#e377c2",
            marker="o",
             markersize=5,
            linewidth=2.0,
            linestyle="-.",
            label="Évaluation déterministe (EvalCallback)",
        )
        if best_ts is not None:
            best_entries = [e for e in eval_hist if e["timestep"] == best_ts]
            if best_entries:
                ax1.scatter(
                    [best_ts],
                    [best_entries[-1]["mean_raw_reward"]],
                    color="#ffd700",
                    edgecolor="black",
                    s=150,
                    zorder=6,
                    marker="*",
                    label=f"Meilleur Checkpoint ({best_ts:,} pas)",
                )

    ax1.axhline(
        random_reward,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Agent aléatoire (hasard : {random_reward:+.1f})",
    )
    ax1.axhline(
        13.0,
        color="#9467bd",
        linestyle=":",
        linewidth=1.6,
        label="Seuil de réussite fiche projet (+13)",
    )
    ax1.set_title(
        "Évolution de la Récompense Cumulée (PPO + EvalCallback)",
        fontsize=12.5,
        fontweight="bold",
    )
    ax1.set_xlabel("Pas d'entraînement (Timesteps)", fontsize=11)
    ax1.set_ylabel("Récompense cumulée (échelle standard)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        ts,
        scores,
        color="#2ca02c",
        alpha=0.18,
        linewidth=0.8,
        label="Tuyaux franchis par épisode",
    )
    ax2.plot(
        ts,
        s_mean,
        color="#2ca02c",
        linewidth=2.4,
        label="Moyenne glissante (entraînement)",
    )
    if eval_hist:
        ev_ts = [e["timestep"] for e in eval_hist]
        ev_s = [e["mean_score"] for e in eval_hist]
        ax2.plot(
            ev_ts,
            ev_s,
            color="#17becf",
            marker="s",
            markersize=5,
            linewidth=2.0,
            linestyle="-.",
            label="Score moyen EvalCallback (déterministe)",
        )
        if best_ts is not None:
            best_entries = [e for e in eval_hist if e["timestep"] == best_ts]
            if best_entries:
                ax2.scatter(
                    [best_ts],
                    [best_entries[-1]["mean_score"]],
                    color="#ffd700",
                    edgecolor="black",
                    s=150,
                    zorder=6,
                    marker="*",
                    label=f"Meilleur Checkpoint ({best_entries[-1]['mean_score']:.1f} tuyaux)",
                )

    ax2.axhline(
        15.0,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.8,
        label="Objectif robustesse (15 tuyaux)",
    )
    ax2.axhline(
        0.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.6,
        label="Agent aléatoire (0 tuyau)",
    )
    ax2.set_title(
        "Nombre de Tuyaux Franchis (Score)", fontsize=12.5, fontweight="bold"
    )
    ax2.set_xlabel("Pas d'entraînement (Timesteps)", fontsize=11)
    ax2.set_ylabel("Score (tuyaux franchis, plafond=100)", fontsize=11)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Projet RL Flappy Bird — Entraînement Poussé PPO (500 000 pas) & Sélection EvalCallback",
        fontsize=14.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_main = os.path.join(out_dir, "courbe_apprentissage_ppo.png")
    plt.savefig(out_main, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Courbe principale sauvegardée : {out_main}")

    # -------------------------------------------------------------------------
    # Figure 2 : Comparaison Bonus (PPO Shaped+EvalCallback vs PPO Default vs DQN)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    for run in loaded_runs:
        ts_r = np.array(run["timesteps"])
        r_m, _ = rolling_mean_std(run["raw_rewards"], window=50)
        ax1.plot(ts_r, r_m, label=run["label"], color=run["color"], linewidth=2.2)

    ax1.axhline(
        random_reward,
        color="#d62728",
        linestyle="--",
        linewidth=1.6,
        label=f"Hasard ({random_reward:+.1f})",
    )
    ax1.set_title(
        "Comparaison des Courbes d'Apprentissage (Récompense brute)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Pas d'entraînement (Timesteps)", fontsize=11)
    ax1.set_ylabel("Récompense brute (moyenne glissante)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(True, alpha=0.3)

    # Bar chart des évaluations finales (sur 30 épisodes déterministes)
    labels = ["Hasard\n(Aléatoire)"] + [
        r["label"]
        .replace(" + EvalCallback (Retenu)", "\n+ EvalCallback")
        .replace(" + ", "\n")
        for r in loaded_runs
    ]
    means = [random_reward] + [r["eval_stats"]["mean_raw_reward"] for r in loaded_runs]
    scores_bar = [0.0] + [r["eval_stats"]["mean_score"] for r in loaded_runs]
    colors = ["#d62728"] + [r["color"] for r in loaded_runs]

    bars = ax2.bar(
        labels, means, color=colors, alpha=0.88, edgecolor="black", width=0.55
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    span = max(means) - min(means)
    for bar, val, sc in zip(bars, means, scores_bar):
        height = bar.get_height()
        offset = 0.03 * span if height >= 0 else -0.08 * span
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f"{val:+.1f}\n({sc:.1f} tuyaux)",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontweight="bold",
            fontsize=9.5,
        )

    ax2.set_title(
        "Performance Finale en Évaluation (30 épisodes déterministes)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Récompense brute moyenne par épisode", fontsize=11)
    ax2.set_ylim(min(means) - 0.18 * span, max(means) + 0.22 * span)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Étude Comparative (Bonus) : PPO + EvalCallback vs PPO Défaut vs DQN",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_comp = os.path.join(out_dir, "comparaison_algos_reward.png")
    plt.savefig(out_comp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Courbe comparative sauvegardée : {out_comp}")


if __name__ == "__main__":
    generate_plots()
