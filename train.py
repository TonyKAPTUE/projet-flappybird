"""
Rôle B — L'agent / l'entraînement poussé (PPO & DQN avec EvalCallback et sauvegarde du meilleur checkpoint)
Module : Apprentissage par renforcement — Projet ISEN Nantes (M2 DP IA)
"""

import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_wrapper import make_flappy_env


class MetricsLoggerCallback(BaseCallback):
    """
    Callback personnalisé pour enregistrer l'évolution des récompenses brutes,
    des récompenses shaped, du score (nombre de tuyaux franchis) et de la
    longueur des épisodes au fil des pas de temps (timesteps).
    Peut reprendre l'historique d'un entraînement précédent si `initial_history` est fourni.
    """

    def __init__(
        self,
        initial_history: Optional[Dict[str, list]] = None,
        timestep_offset: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.timestep_offset = timestep_offset
        if initial_history is not None:
            self.timesteps: List[int] = list(initial_history.get("timesteps", []))
            self.raw_rewards: List[float] = list(initial_history.get("raw_rewards", []))
            self.shaped_rewards: List[float] = list(
                initial_history.get("shaped_rewards", [])
            )
            self.scores: List[int] = list(initial_history.get("scores", []))
            self.lengths: List[int] = list(initial_history.get("lengths", []))
        else:
            self.timesteps = []
            self.raw_rewards = []
            self.shaped_rewards = []
            self.scores = []
            self.lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_raw_reward" in info:
                self.timesteps.append(int(self.timestep_offset + self.num_timesteps))
                self.raw_rewards.append(float(info["episode_raw_reward"]))
                self.shaped_rewards.append(float(info["episode_shaped_reward"]))
                self.scores.append(int(info["episode_score"]))
                self.lengths.append(int(info["episode_length"]))
        return True

    def to_dict(self) -> Dict[str, list]:
        return {
            "timesteps": self.timesteps,
            "raw_rewards": self.raw_rewards,
            "shaped_rewards": self.shaped_rewards,
            "scores": self.scores,
            "lengths": self.lengths,
        }


def evaluate_agent(
    model,
    n_episodes: int = 30,
    reward_mode: str = "default",
    seed: int = 100,
    score_limit: int = 100,
) -> Dict[str, float]:
    """
    Évalue un agent (ou une politique aléatoire si model=None) sur n_episodes
    déterministes et renvoie les statistiques complètes de performance.
    """
    env = make_flappy_env(
        render_mode=None, reward_mode=reward_mode, score_limit=score_limit
    )
    raw_rewards = []
    scores = []
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        raw_rewards.append(info["episode_raw_reward"])
        scores.append(info["episode_score"])
        lengths.append(info["episode_length"])

    env.close()
    scores_arr = np.array(scores)
    return {
        "mean_raw_reward": float(np.mean(raw_rewards)),
        "std_raw_reward": float(np.std(raw_rewards)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": int(np.min(scores)),
        "max_score": int(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
        "pass_at_least_1_rate": float(np.mean(scores_arr >= 1) * 100.0),
        "pass_at_least_5_rate": float(np.mean(scores_arr >= 5) * 100.0),
        "pass_at_least_10_rate": float(np.mean(scores_arr >= 10) * 100.0),
        "pass_at_least_15_rate": float(np.mean(scores_arr >= 15) * 100.0),
    }


class BestCheckpointEvalCallback(BaseCallback):
    """
    Callback d'évaluation périodique complémentaire à `EvalCallback` permettant de :
      - mesurer en cours d'entraînement le score exact en nombre de tuyaux franchis
        et la récompense brute standard (sans biais de Reward Shaping),
      - sauvegarder automatiquement le meilleur "checkpoint" dès que le score moyen
        (ou le score minimum sur les parties les plus difficiles) bat le record.
    """

    def __init__(
        self,
        eval_freq_steps: int,
        best_model_save_path: str,
        n_eval_episodes: int = 15,
        score_limit: int = 100,
        timestep_offset: int = 0,
        initial_eval_history: Optional[List[Dict]] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq_steps = eval_freq_steps
        self.best_model_save_path = best_model_save_path
        self.n_eval_episodes = n_eval_episodes
        self.score_limit = score_limit
        self.timestep_offset = timestep_offset
        self.eval_history: List[Dict] = (
            list(initial_eval_history) if initial_eval_history else []
        )
        self.best_mean_score = -np.inf
        self.best_min_score = -1
        self.best_mean_raw_reward = -np.inf
        self.best_timestep = 0
        self._last_eval_num_timesteps = 0

    def _run_evaluation(self, current_ts: int) -> None:
        stats = evaluate_agent(
            self.model,
            n_episodes=self.n_eval_episodes,
            reward_mode="default",
            seed=100,
            score_limit=self.score_limit,
        )
        entry = {
            "timestep": int(current_ts),
            "mean_score": stats["mean_score"],
            "min_score": stats["min_score"],
            "max_score": stats["max_score"],
            "mean_raw_reward": stats["mean_raw_reward"],
            "pass_at_least_10_rate": stats["pass_at_least_10_rate"],
            "pass_at_least_15_rate": stats["pass_at_least_15_rate"],
        }
        self.eval_history.append(entry)

        is_better = (
            (stats["mean_score"] > self.best_mean_score)
            or (
                np.isclose(stats["mean_score"], self.best_mean_score)
                and stats["min_score"] > self.best_min_score
            )
            or (
                np.isclose(stats["mean_score"], self.best_mean_score)
                and stats["min_score"] == self.best_min_score
                and stats["mean_raw_reward"] > self.best_mean_raw_reward
            )
        )

        marker = ""
        if is_better:
            self.best_mean_score = stats["mean_score"]
            self.best_min_score = stats["min_score"]
            self.best_mean_raw_reward = stats["mean_raw_reward"]
            self.best_timestep = int(current_ts)
            self.model.save(self.best_model_save_path)
            marker = " ★ NOUVEAU MEILLEUR CHECKPOINT SAUVEGARDÉ !"

        if self.verbose >= 1:
            print(
                f"  [EvalCallback @ {current_ts:7,d} pas] "
                f"Score moyen = {stats['mean_score']:6.2f} (min={stats['min_score']:3d}, max={stats['max_score']:3d}) | "
                f" >=15 tuyaux = {stats['pass_at_least_15_rate']:5.1f}% | "
                f"Récompense brute = {stats['mean_raw_reward']:+7.2f}{marker}"
            )

    def _on_training_start(self) -> None:
        current_ts = self.timestep_offset + self.num_timesteps
        self._run_evaluation(current_ts)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_num_timesteps) >= self.eval_freq_steps:
            self._last_eval_num_timesteps = self.num_timesteps
            current_ts = self.timestep_offset + self.num_timesteps
            self._run_evaluation(current_ts)
        return True


def train_model(
    algo: str = "ppo",
    reward_mode: str = "shaped",
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    continue_if_exists: bool = True,
) -> str:
    """
    Entraîne (ou pousse l'entraînement d') un modèle PPO ou DQN sur FlappyBird-v0
    (use_lidar=False) avec EvalCallback pour sauvegarder automatiquement le
    meilleur checkpoint en cours d'entraînement.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run_name = f"flappy_{algo.lower()}_{reward_mode}"
    model_path = os.path.join(save_dir, run_name)
    best_checkpoint_path = os.path.join(save_dir, f"best_{run_name}")
    sb3_eval_dir = os.path.join(save_dir, f"eval_sb3_{run_name}")
    log_path = os.path.join(log_dir, f"{run_name}_metrics.json")

    def make_env_fn(rank: int):
        def _init():
            env = make_flappy_env(
                render_mode=None, reward_mode=reward_mode, score_limit=100
            )
            env.reset(seed=seed + rank)
            return env

        return _init

    vec_env = DummyVecEnv([make_env_fn(i) for i in range(n_envs)])

    # Environnement d'évaluation dédié pour le EvalCallback officiel de SB3
    eval_env_sb3 = DummyVecEnv(
        [
            lambda: Monitor(
                make_flappy_env(
                    render_mode=None, reward_mode=reward_mode, score_limit=50
                )
            )
        ]
    )

    initial_history = None
    timestep_offset = 0
    steps_to_run = total_timesteps
    elapsed_prev = 0.0

    # Si un modèle et son historique existent déjà, on peut pousser l'entraînement plus loin
    if (
        continue_if_exists
        and os.path.exists(model_path + ".zip")
        and os.path.exists(log_path)
    ):
        with open(log_path, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        prev_ts = int(prev_data.get("total_timesteps", 0))
        if prev_ts < total_timesteps:
            initial_history = prev_data
            timestep_offset = prev_ts
            steps_to_run = total_timesteps - prev_ts
            elapsed_prev = float(prev_data.get("training_time_sec", 0.0))

    print(f"\n============================================================")
    if timestep_offset > 0:
        print(
            f"Poussée de l'entraînement : {run_name} | "
            f"{timestep_offset:,} -> {total_timesteps:,} pas (+{steps_to_run:,} pas)"
        )
    else:
        print(f"Entraînement complet : {run_name} | Timesteps : {total_timesteps:,}")
    print(f"============================================================")

    if algo.lower() == "ppo":
        if timestep_offset > 0:
            model = PPO.load(model_path, env=vec_env, seed=seed)
            # Fine-tuning doux pour stabiliser la politique au sommet sans dégrader le checkpoint
            model.learning_rate = 5e-5
            model.ent_coef = 0.001
            model._setup_lr_schedule()
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=2.5e-4,
                n_steps=512,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                policy_kwargs=dict(net_arch=[128, 128]),
                verbose=1,
                seed=seed,
            )
    elif algo.lower() == "dqn":
        if timestep_offset > 0:
            model = DQN.load(model_path, env=vec_env, seed=seed)
        else:
            model = DQN(
                "MlpPolicy",
                vec_env,
                learning_rate=2.5e-4,
                buffer_size=100_000,
                learning_starts=2_000,
                batch_size=128,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1_000,
                exploration_fraction=0.25,
                exploration_final_eps=0.02,
                policy_kwargs=dict(net_arch=[128, 128]),
                verbose=1,
                seed=seed,
            )
    else:
        raise ValueError(f"Algorithme non supporté: {algo}")

    # 1) Callback de suivi des épisodes d'entraînement
    metrics_cb = MetricsLoggerCallback(
        initial_history=initial_history, timestep_offset=timestep_offset
    )

    # 2) EvalCallback officiel de Stable-Baselines3 (sauvegarde best_model.zip)
    sb3_eval_cb = EvalCallback(
        eval_env_sb3,
        best_model_save_path=sb3_eval_dir,
        log_path=sb3_eval_dir,
        eval_freq=max(10_000, steps_to_run // (3 * n_envs)),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # 3) Callback d'évaluation complète avec sélection du meilleur checkpoint sur le score (tuyaux)
    best_ckpt_cb = BestCheckpointEvalCallback(
        eval_freq_steps=max(30_000, steps_to_run // 4),
        best_model_save_path=best_checkpoint_path,
        n_eval_episodes=8,
        score_limit=100,
        timestep_offset=timestep_offset,
        initial_eval_history=(
            initial_history.get("eval_history", None) if initial_history else None
        ),
        verbose=1,
    )

    callbacks = CallbackList([metrics_cb, sb3_eval_cb, best_ckpt_cb])

    start_time = time.time()
    model.learn(total_timesteps=steps_to_run, callback=callbacks)
    elapsed = elapsed_prev + (time.time() - start_time)

    # Évaluation de fin d'entraînement
    best_ckpt_cb._run_evaluation(total_timesteps)

    # Restauration automatique du MEILLEUR checkpoint sauvegardé par EvalCallback !
    if os.path.exists(best_checkpoint_path + ".zip"):
        print(
            f"\n[EvalCallback] Sélection du meilleur checkpoint "
            f"(obtenu à {best_ckpt_cb.best_timestep:,} pas : "
            f"score moyen = {best_ckpt_cb.best_mean_score:.2f}, min = {best_ckpt_cb.best_min_score})"
        )
        shutil.copyfile(best_checkpoint_path + ".zip", model_path + ".zip")
        model = (
            PPO.load(best_checkpoint_path)
            if algo.lower() == "ppo"
            else DQN.load(best_checkpoint_path)
        )
    else:
        model.save(model_path)

    # Sauvegarde additionnelle sous "flappy_ppo.zip" si c'est le modèle PPO principal
    if algo.lower() == "ppo" and reward_mode == "shaped":
        model.save("flappy_ppo")

    eval_stats = evaluate_agent(
        model, n_episodes=30, reward_mode=reward_mode, score_limit=100
    )
    print(f"\n[Résultats Évaluation Finale (30 épisodes) — {run_name} ({elapsed:.1f}s)]")
    print(
        f"  Récompense brute moyenne : {eval_stats['mean_raw_reward']:+.2f} ± {eval_stats['std_raw_reward']:.2f}"
    )
    print(
        f"  Tuyaux franchis (Score)  : {eval_stats['mean_score']:.2f} ± {eval_stats['std_score']:.2f} "
        f"(min={eval_stats['min_score']}, max={eval_stats['max_score']})"
    )
    print(
        f"  Taux >= 1 tuyau          : {eval_stats['pass_at_least_1_rate']:.1f}% | "
        f"Taux >= 5 tuyaux  : {eval_stats['pass_at_least_5_rate']:.1f}%"
    )
    print(
        f"  Taux >= 10 tuyaux        : {eval_stats['pass_at_least_10_rate']:.1f}% | "
        f"Taux >= 15 tuyaux : {eval_stats['pass_at_least_15_rate']:.1f}%"
    )

    log_data = metrics_cb.to_dict()
    log_data["algo"] = algo.lower()
    log_data["reward_mode"] = reward_mode
    log_data["total_timesteps"] = total_timesteps
    log_data["training_time_sec"] = elapsed
    log_data["best_checkpoint_timestep"] = best_ckpt_cb.best_timestep
    log_data["eval_history"] = best_ckpt_cb.eval_history
    log_data["eval_stats"] = eval_stats

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    vec_env.close()
    eval_env_sb3.close()
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement RL Flappy Bird avec EvalCallback"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "all"],
        help="Algorithme à entraîner (ppo, dqn ou all pour tout ré-entraîner)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Budget total de pas d'entraînement pour PPO principal (défaut: 500 000)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Repartir de zéro au lieu de pousser l'entraînement du checkpoint existant",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    # 1. Évaluation de la baseline aléatoire (hasard)
    if not os.path.exists("logs/random_baseline.json") or args.from_scratch:
        print("Évaluation de l'agent aléatoire (hasard)...")
        random_stats = evaluate_agent(None, n_episodes=50, reward_mode="default")
        print(
            f"  [Hasard] Récompense moyenne = {random_stats['mean_raw_reward']:+.2f} | "
            f"Score moyen = {random_stats['mean_score']:.2f}"
        )
        with open("logs/random_baseline.json", "w", encoding="utf-8") as f:
            json.dump(random_stats, f, indent=2)

    if args.algo == "ppo":
        train_model(
            "ppo",
            "shaped",
            total_timesteps=args.timesteps,
            continue_if_exists=not args.from_scratch,
        )
    elif args.algo == "dqn":
        train_model(
            "dqn",
            "shaped",
            total_timesteps=args.timesteps,
            continue_if_exists=not args.from_scratch,
        )
    else:
        train_model(
            "ppo",
            "shaped",
            total_timesteps=args.timesteps,
            continue_if_exists=not args.from_scratch,
        )
        train_model(
            "ppo",
            "default",
            total_timesteps=250_000,
            continue_if_exists=not args.from_scratch,
        )
        train_model(
            "dqn",
            "shaped",
            total_timesteps=250_000,
            continue_if_exists=not args.from_scratch,
        )
