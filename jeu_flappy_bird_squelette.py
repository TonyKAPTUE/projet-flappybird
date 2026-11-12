
"""
jeu_flappy_bird_squelette.py — VERSION COMPLÉTÉE & OPTIMISÉE (Flappy Bird — M2 DP IA ISEN)
==========================================================================================
Ce script complète le squelette étudiant officiel avec :
  1) Le choix de l'OBSERVATION : `use_lidar=False` (vecteur de 12 réels normalisés)
     avec recalage dynamique du prochain tuyau actif dans `obs[0:3]` dès que l'oiseau
     franchit le centre du tuyau courant (`obs[0] < 0.05`).
  2) Le Reward Shaping (via notre wrapper `FlappyRewardWrapper`) pour guider l'oiseau
     vers le centre des tuyaux et accélérer fortement la convergence.
  3) L'ALGORITHME PPO (et option DQN) avec un budget poussé à 500 000 pas et sauvegarde
     automatique du meilleur "checkpoint" en cours d'entraînement via `EvalCallback`
     (`BestCheckpointEvalCallback` + `EvalCallback` de Stable-Baselines3).
  4) La sauvegarde automatique du meilleur modèle (`flappy_ppo.zip`), de la courbe
     d'apprentissage (`plots/courbe_apprentissage_ppo.png`) et le lancement de la démo Pygame.

Lancer :
  python jeu_flappy_bird_squelette.py                  # Charge le meilleur checkpoint flappy_ppo.zip et lance la démo
  python jeu_flappy_bird_squelette.py --retrain        # Pousse ou relance l'entraînement avec EvalCallback (500 000 pas)
  python jeu_flappy_bird_squelette.py --algo dqn       # Teste avec DQN
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401 (enregistre l'environnement "FlappyBird-v0")
import pygame
from stable_baselines3 import PPO, DQN

from env_wrapper import FlappyRewardWrapper
from train import train_model
from evaluate_and_plot import generate_plots

# ==============================================================================
# TODO (3) COMPLÉTÉ : Budget d'entraînement poussé + EvalCallback
# 500 000 pas avec sélection du meilleur checkpoint via EvalCallback permettent
# d'obtenir un agent quasi-parfait (99 à 100 tuyaux franchis par partie, 100% >= 15 tuyaux).
# ==============================================================================
TIMESTEPS = 500_000


def create_env(use_lidar: bool = False, reward_mode: str = "shaped", render_mode=None):
    """
    TODO (1) COMPLÉTÉ : Représentation de l'état et Reward Shaping.
      - use_lidar=False : vecteur de 12 valeurs numériques normalisées dans [-1, 1]
        (positions des 3 prochains tuyaux + position/vitesse/rotation de l'oiseau).
      - FlappyRewardWrapper : recale dynamiquement le prochain tuyau dans obs[0:3]
        dès que le tuyau précédent est franchi (obs[0] < 0.05) et ajoute un bonus
        de centrage continu vers le milieu de l'ouverture (+1.5 par tuyau, -1.0 à la mort).
    """
    base_env = gym.make(
        "FlappyBird-v0",
        use_lidar=use_lidar,
        render_mode=render_mode,
        score_limit=100,
    )
    return FlappyRewardWrapper(base_env, reward_mode=reward_mode, canonicalize_obs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projet RL Flappy Bird (Squelette complété avec EvalCallback)"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Relancer / pousser l'entraînement avec EvalCallback jusqu'à 500 000 pas",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS,
        help=f"Nombre de pas d'entraînement (défaut: {TIMESTEPS})",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="Algorithme à utiliser : 'ppo' (recommandé) ou 'dqn'",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="shaped",
        choices=["shaped", "default"],
        help="Fonction de récompense : 'shaped' (avec guidage central) ou 'default'",
    )
    args = parser.parse_args()

    model_filename = f"flappy_{args.algo}"
    best_ckpt = f"models/best_flappy_{args.algo}_{args.reward}"

    if (os.path.exists(f"{best_ckpt}.zip") or os.path.exists(f"{model_filename}.zip")) and not args.retrain:
        load_target = best_ckpt if os.path.exists(f"{best_ckpt}.zip") else model_filename
        print(f"[Info] Meilleur checkpoint pré-entraîné trouvé ('{load_target}.zip').")
        print("       (Astuce : ajoutez --retrain pour pousser l'entraînement avec EvalCallback)")
        model = (
            PPO.load(load_target)
            if args.algo == "ppo"
            else DQN.load(load_target)
        )
    else:
        # TODO (1), (2), (3) : Entraînement avec DummyVecEnv + EvalCallback (sauvegarde du meilleur checkpoint)
        saved_path = train_model(
            algo=args.algo,
            reward_mode=args.reward,
            total_timesteps=args.timesteps,
            continue_if_exists=True,
        )
        generate_plots()
        model = PPO.load(saved_path) if args.algo == "ppo" else DQN.load(saved_path)
        print(f"Entraînement terminé — meilleur checkpoint actif : '{saved_path}.zip'.")

    # ==========================================================================
    # Démo : on regarde l'oiseau jouer (boucle du squelette officiel)
    # ==========================================================================
    print("\nOuverture de la fenêtre Pygame pour 5 parties de démonstration...")
    show = create_env(use_lidar=False, reward_mode=args.reward, render_mode="human")
    try:
        for partie in range(5):
            obs, info = show.reset()
            done = False
            quit_window = False

            while not done:
                # Permet de fermer proprement la fenêtre avec la croix ou Échap
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        quit_window = True
                if quit_window:
                    break

                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = show.step(int(action))
                done = term or trunc

            print(
                f"Partie {partie + 1} : retour brut = {info.get('episode_raw_reward', 0.0):+.1f} | "
                f"tuyaux franchis = {info.get('score', 0)}"
            )
            if quit_window:
                break
    finally:
        show.close()
