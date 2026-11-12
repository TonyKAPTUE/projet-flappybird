"""
Rôle A — Le jeu / l'environnement (Flappy Bird Gymnasium + Reward Shaping & Canonical Obs)
Module : Apprentissage par renforcement — Projet ISEN Nantes (M2 DP IA)
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
import flappy_bird_gymnasium  # noqa: F401 (enregistre FlappyBird-v0)


class FlappyRewardWrapper(gym.Wrapper):
    """
    Wrapper Gymnasium autour de FlappyBird-v0 (use_lidar=False) permettant :
      1) De recaler dynamiquement la fenêtre des 3 tuyaux dans le vecteur d'observation
         (12 réels) dès que le centre de l'oiseau a franchi le tuyau courant (obs[0] < 0.05),
         afin que `obs[0:3]` désigne TOUJOURS le prochain tuyau à franchir et non un tuyau
         déjà derrière l'oiseau.
      2) D'appliquer différentes fonctions de récompense (Reward Shaping) tout en
         conservant le suivi de la récompense brute (standard) pour une comparaison
         équitable entre les agents.

    Rappel de l'état (12 réels normalisés dans [-1, 1]) :
      0: pos_x du prochain tuyau actif (normalisé par screen_width=288)
      1: pos_y du bas du tuyau supérieur actif (normalisé par screen_height=512)
      2: pos_y du haut du tuyau inférieur actif (normalisé par screen_height=512)
      3: pos_x du tuyau suivant
      4: pos_y du bas du tuyau supérieur suivant
      5: pos_y du haut du tuyau inférieur suivant
      6: pos_x du troisième tuyau
      7: pos_y du bas du troisième tuyau supérieur
      8: pos_y du haut du troisième tuyau inférieur
      9: pos_y de l'oiseau (normalisé par screen_height=512)
     10: vel_y de l'oiseau (normalisé par max_vel_y=10)
     11: rotation de l'oiseau (normalisé par 90°)
    """

    # Seuil horizontal normalisé (x_pipe / 288) en dessous duquel le centre de l'oiseau
    # (x_center = 74 px) a dépassé le bord droit du tuyau (x_pipe + 52 px < 66.4 px).
    PIPE_PASSED_X_THRESHOLD = 0.05

    def __init__(
        self,
        env: gym.Env,
        reward_mode: str = "shaped",
        centering_coef: float = 0.05,
        pipe_bonus: float = 1.5,
        death_penalty: float = -1.0,
        canonicalize_obs: bool = True,
    ) -> None:
        super().__init__(env)
        assert reward_mode in ("default", "shaped"), (
            f"reward_mode inconnu: {reward_mode}. Choisir 'default' ou 'shaped'."
        )
        self.reward_mode = reward_mode
        self.centering_coef = centering_coef
        self.pipe_bonus = pipe_bonus
        self.death_penalty = death_penalty
        self.canonicalize_obs = canonicalize_obs

        # Statistiques par épisode (pour le monitoring et EvalCallback)
        self._ep_raw_reward = 0.0
        self._ep_shaped_reward = 0.0
        self._ep_length = 0
        self._last_score = 0

    def _canonicalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Dans FlappyBird-v0 brut, lorsqu'un tuyau est dépassé par l'oiseau (x < 0.05),
        il reste dans `obs[0:3]` jusqu'à sa sortie complète de l'écran à gauche (x = -0.18),
        c'est-à-dire jusqu'au moment où l'oiseau touche déjà le tuyau suivant (x = 0.32).
        Ce recalage place systématiquement le prochain tuyau à franchir dans `obs[0:3]`.
        """
        if self.canonicalize_obs and obs[0] < self.PIPE_PASSED_X_THRESHOLD:
            canon = obs.copy()
            canon[0:3] = obs[3:6]
            canon[3:6] = obs[6:9]
            canon[6:9] = [1.0, 0.0, 1.0]
            return canon
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._ep_raw_reward = 0.0
        self._ep_shaped_reward = 0.0
        self._ep_length = 0
        self._last_score = info.get("score", 0)
        return self._canonicalize_observation(obs), info

    def _get_target_gap_center_y(self, obs: np.ndarray) -> float:
        """
        Calcule la coordonnée Y normalisée du centre de l'ouverture du prochain tuyau.
        Si `canonicalize_obs=True`, `obs[0:3]` est déjà le prochain tuyau actif.
        """
        if not self.canonicalize_obs and obs[0] < self.PIPE_PASSED_X_THRESHOLD:
            top_y, bottom_y = obs[4], obs[5]
        else:
            top_y, bottom_y = obs[1], obs[2]
        return 0.5 * (top_y + bottom_y)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_obs, raw_reward, terminated, truncated, info = self.env.step(action)
        obs = self._canonicalize_observation(raw_obs)

        self._ep_raw_reward += float(raw_reward)
        self._ep_length += 1

        current_score = info.get("score", 0)
        passed_pipe = current_score > self._last_score
        self._last_score = current_score

        if self.reward_mode == "default":
            shaped_reward = float(raw_reward)
        else:
            # --- Reward Shaping ---
            # 1) Mort : pénalité franche
            if terminated:
                shaped_reward = self.death_penalty
            # 2) Tuyau franchi : bonus renforcé
            elif passed_pipe:
                shaped_reward = self.pipe_bonus
            # 3) Survie : +0.1 de base (ou -0.5 si touche le plafond y < 0)
            else:
                shaped_reward = float(raw_reward)

            # 4) Guidage continu vers le centre de l'ouverture du prochain tuyau
            if not terminated:
                bird_y = obs[9]
                gap_center_y = self._get_target_gap_center_y(obs)
                dist_to_center = abs(bird_y - gap_center_y)
                # Récompense additionnelle dans [-centering_coef, +centering_coef]
                # Positive quand l'oiseau est proche du centre (dist < 0.25),
                # négative quand il s'en éloigne fortement.
                centering_bonus = self.centering_coef * (1.0 - 4.0 * dist_to_center)
                shaped_reward += float(centering_bonus)

        self._ep_shaped_reward += shaped_reward

        if terminated or truncated:
            info["episode_raw_reward"] = self._ep_raw_reward
            info["episode_shaped_reward"] = self._ep_shaped_reward
            info["episode_score"] = current_score
            info["episode_length"] = self._ep_length

        return obs, shaped_reward, terminated, truncated, info


def make_flappy_env(
    render_mode: Optional[str] = None,
    reward_mode: str = "shaped",
    use_lidar: bool = False,
    score_limit: Optional[int] = 100,
    canonicalize_obs: bool = True,
) -> gym.Env:
    """
    Crée et configure l'environnement FlappyBird-v0 avec état vectoriel (12 réels)
    et le wrapper de Reward Shaping + recalage d'observation.
    """
    env = gym.make(
        "FlappyBird-v0",
        use_lidar=use_lidar,
        render_mode=render_mode,
        score_limit=score_limit,
    )
    env = FlappyRewardWrapper(
        env, reward_mode=reward_mode, canonicalize_obs=canonicalize_obs
    )
    return env


if __name__ == "__main__":
    # Vérification rapide de l'environnement (Rôle A)
    env = make_flappy_env(render_mode=None, reward_mode="shaped")
    obs, info = env.reset(seed=42)
    print("=== Vérification de l'environnement FlappyBird-v0 ===")
    print(f"Espace d'observation : {env.observation_space} (shape={obs.shape})")
    print(f"Espace d'actions     : {env.action_space} (0=Ne rien faire, 1=Battre des ailes)")
    print(f"Observation initiale : {np.round(obs, 3)}")

    total_shaped = 0.0
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_shaped += reward
        steps += 1
        done = terminated or truncated

    print(
        f"Épisode aléatoire terminé en {steps} pas | "
        f"Score (tuyaux) = {info['score']} | "
        f"Récompense brute = {info['episode_raw_reward']:.2f} | "
        f"Récompense shaped = {info['episode_shaped_reward']:.2f}"
    )
    env.close()
