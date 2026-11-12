"""
Rôle C — Démo « Regarder jouer » (Interface graphique Pygame)
Module : Apprentissage par renforcement — Projet ISEN Nantes (M2 DP IA)

Exemples d'utilisation :
  python demo.py                                 # Joue avec le meilleur checkpoint PPO (fenêtre Pygame)
  python demo.py --algo dqn                      # Joue avec le modèle DQN
  python demo.py --random                        # Montre un oiseau qui joue au hasard (pour comparer)
  python demo.py --episodes 3 --fps 30           # Règle le nombre d'épisodes et la vitesse
"""

import argparse
import os
import time

import pygame
from stable_baselines3 import DQN, PPO

from env_wrapper import make_flappy_env


def load_trained_model(algo: str = "ppo", model_path: str = ""):
    if model_path and os.path.exists(model_path + ".zip"):
        return PPO.load(model_path) if algo == "ppo" else DQN.load(model_path)

    candidates = [
        f"models/best_flappy_{algo}_shaped",
        f"models/flappy_{algo}_shaped",
        f"models/flappy_{algo}_default",
        "flappy_ppo",
    ]
    for cand in candidates:
        if os.path.exists(cand + ".zip"):
            print(f"[Démo] Chargement du meilleur modèle : {cand}.zip")
            return PPO.load(cand) if algo == "ppo" else DQN.load(cand)

    raise FileNotFoundError(
        "Aucun modèle entraîné trouvé. Lancez d'abord `python train.py`."
    )


def run_demo(
    algo: str = "ppo",
    use_random: bool = False,
    episodes: int = 3,
    fps: int = 30,
    max_steps_per_ep: int = 4000,
    model_path: str = "",
) -> None:
    model = None
    if not use_random:
        model = load_trained_model(algo=algo, model_path=model_path)

    # Création de l'environnement avec render_mode="human" comme demandé sur la fiche
    env = make_flappy_env(
        render_mode="human", reward_mode="shaped", use_lidar=False, score_limit=100
    )
    env.unwrapped.metadata["render_fps"] = fps

    mode_str = (
        "ALÉATOIRE (Hasard)"
        if use_random
        else f"AGENT {algo.upper()} ENTRAÎNÉ (Best Checkpoint EvalCallback)"
    )
    print(f"\n=======================================================")
    print(f" DÉMO FLAPPY BIRD — Mode : {mode_str}")
    print(f" Fermez la fenêtre Pygame ou appuyez sur ÉCHAP pour quitter.")
    print(f"=======================================================\n")

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            steps = 0
            quit_requested = False

            while not done and steps < max_steps_per_ep:
                # Gestion de la fermeture propre de la fenêtre Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        quit_requested = True

                if quit_requested:
                    break

                if use_random:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)

                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated

            print(
                f"Épisode {ep}/{episodes} | Pas survécus : {steps:4d} | "
                f"Tuyaux franchis (Score) : {info.get('score', 0):3d} | "
                f"Récompense brute : {info.get('episode_raw_reward', 0.0):+.2f}"
            )
            if quit_requested:
                print("Arrêt demandé par l'utilisateur.")
                break
            time.sleep(0.6)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démo visuelle Flappy Bird RL")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="Algorithme à démontrer (ppo ou dqn)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Jouer au hasard (sans modèle) pour comparer",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Nombre de parties à jouer"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Images par seconde (défaut: 30)"
    )
    parser.add_argument(
        "--model-path", type=str, default="", help="Chemin spécifique vers un modèle"
    )
    args = parser.parse_args()

    run_demo(
        algo=args.algo,
        use_random=args.random,
        episodes=args.episodes,
        fps=args.fps,
        model_path=args.model_path,
    )
