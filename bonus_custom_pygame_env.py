"""
BONUS (Section 8 — Version complète / Vibe Coding) :
Environnement Flappy Bird codé de zéro en Pygame + Wrapper Gymnasium complet (`CustomFlappyBirdEnv`).

Ce fichier répond au défi bonus de la fiche projet :
« Coder soi-même l'environnement Flappy en pygame et le transformer en env Gymnasium ».
Il implémente la physique complète (gravité, battement d'ailes, défilement des tuyaux,
collisions, état normalisé à 12 réels compatible avec le modèle PPO/DQN, et rendu Pygame).
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class CustomFlappyBirdEnv(gym.Env):
    """
    Environnement Gymnasium Flappy Bird codé intégralement from scratch avec Pygame.
    Produit un vecteur d'état de 12 réels dans [-1, 1] :
      [p0_x, p0_top, p0_bot, p1_x, p1_top, p1_bot, p2_x, p2_top, p2_bot, bird_y, bird_vel, bird_rot]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    SCREEN_W = 288
    SCREEN_H = 512
    GROUND_Y = int(512 * 0.79)  # 404
    BIRD_X = int(288 * 0.2)     # 57
    BIRD_W = 34
    BIRD_H = 24
    PIPE_W = 52
    PIPE_GAP = 100
    PIPE_VEL_X = -4
    GRAVITY = 1.0
    FLAP_VEL = -9.0
    MAX_VEL_Y = 10.0

    def __init__(
        self,
        render_mode: Optional[str] = None,
        score_limit: int = 100,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.score_limit = score_limit

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float64
        )

        self._display = None
        self._clock = None
        self._surface = None
        self._font = None

        self.bird_y = 0.0
        self.bird_vel_y = 0.0
        self.bird_rot = 0.0
        self.score = 0
        self.steps = 0
        self.pipes = []

    def _spawn_pipe(self, x_pos: float) -> Dict[str, float]:
        gap_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        gap_y = float(self.np_random.choice(gap_ys)) + int(self.GROUND_Y * 0.2)
        return {
            "x": float(x_pos),
            "top_y": gap_y,
            "bottom_y": gap_y + self.PIPE_GAP,
            "passed": False,
        }

    def _get_obs(self) -> np.ndarray:
        sorted_pipes = sorted(self.pipes, key=lambda p: p["x"])
        feats = []
        for p in sorted_pipes[:3]:
            if p["x"] > self.SCREEN_W:
                feats.extend([1.0, 0.0, 1.0])
            else:
                feats.extend([
                    p["x"] / self.SCREEN_W,
                    p["top_y"] / self.SCREEN_H,
                    p["bottom_y"] / self.SCREEN_H,
                ])
        feats.extend([
            self.bird_y / self.SCREEN_H,
            np.clip(self.bird_vel_y / self.MAX_VEL_Y, -1.0, 1.0),
            np.clip(self.bird_rot / 90.0, -1.0, 1.0),
        ])
        return np.array(feats, dtype=np.float64)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.bird_y = float((self.SCREEN_H - self.BIRD_H) // 2)
        self.bird_vel_y = self.FLAP_VEL
        self.bird_rot = 45.0
        self.score = 0
        self.steps = 0

        self.pipes = [
            self._spawn_pipe(self.SCREEN_W),
            self._spawn_pipe(self.SCREEN_W * 1.5),
            self._spawn_pipe(self.SCREEN_W * 2.0),
        ]

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {"score": self.score}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        flapped = False
        if action == 1 and self.bird_y > -2 * self.BIRD_H:
            self.bird_vel_y = self.FLAP_VEL
            self.bird_rot = 45.0
            flapped = True

        if not flapped:
            self.bird_vel_y = min(self.bird_vel_y + self.GRAVITY, self.MAX_VEL_Y)
            self.bird_rot = max(self.bird_rot - 3.0, -90.0)

        self.bird_y += min(
            self.bird_vel_y, self.GROUND_Y - self.bird_y - self.BIRD_H
        )

        reward = 0.1
        bird_mid_x = self.BIRD_X + self.BIRD_W / 2.0

        for p in self.pipes:
            p["x"] += self.PIPE_VEL_X
            pipe_mid_x = p["x"] + self.PIPE_W / 2.0
            if not p["passed"] and pipe_mid_x <= bird_mid_x < pipe_mid_x + 4:
                p["passed"] = True
                self.score += 1
                reward = 1.5

        # Recyclage des tuyaux sortis à gauche
        if self.pipes[0]["x"] < -self.PIPE_W:
            self.pipes.pop(0)
            new_x = self.SCREEN_W + self.PIPE_W + (self.SCREEN_W * 0.2)
            self.pipes.append(self._spawn_pipe(new_x))

        # Pénalité plafond
        if self.bird_y < 0:
            reward = -0.5

        # Collision sol ou tuyau
        terminated = False
        if self.bird_y + self.BIRD_H >= self.GROUND_Y - 1:
            terminated = True
            reward = -1.0
        else:
            bird_rect = pygame.Rect(
                int(self.BIRD_X), int(self.bird_y), self.BIRD_W, self.BIRD_H
            )
            for p in self.pipes:
                up_rect = pygame.Rect(int(p["x"]), 0, self.PIPE_W, int(p["top_y"]))
                low_rect = pygame.Rect(
                    int(p["x"]),
                    int(p["bottom_y"]),
                    self.PIPE_W,
                    self.GROUND_Y - int(p["bottom_y"]),
                )
                if bird_rect.colliderect(up_rect) or bird_rect.colliderect(low_rect):
                    terminated = True
                    reward = -1.0
                    break

        truncated = self.score >= self.score_limit
        obs = self._get_obs()

        # Reward shaping continu (centrage dans le prochain tuyau)
        if not terminated:
            target_p = self.pipes[0] if self.pipes[0]["x"] + self.PIPE_W > self.BIRD_X else self.pipes[1]
            gap_center = 0.5 * (target_p["top_y"] + target_p["bottom_y"]) / self.SCREEN_H
            dist = abs((self.bird_y / self.SCREEN_H) - gap_center)
            reward += float(0.05 * (1.0 - 4.0 * dist))

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, {"score": self.score}

    def render(self):
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
            self._font = pygame.font.SysFont("Arial", 26, bold=True)
            if self.render_mode == "human":
                self._display = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
                pygame.display.set_caption("Custom Flappy Bird (Bonus Vibe Coding)")
                self._clock = pygame.time.Clock()

        # Ciel
        self._surface.fill((112, 197, 206))

        # Tuyaux
        for p in self.pipes:
            x = int(p["x"])
            top_h = int(p["top_y"])
            bot_y = int(p["bottom_y"])
            pygame.draw.rect(self._surface, (78, 192, 46), (x, 0, self.PIPE_W, top_h))
            pygame.draw.rect(self._surface, (45, 115, 25), (x, 0, self.PIPE_W, top_h), 2)
            pygame.draw.rect(
                self._surface,
                (78, 192, 46),
                (x, bot_y, self.PIPE_W, self.GROUND_Y - bot_y),
            )
            pygame.draw.rect(
                self._surface,
                (45, 115, 25),
                (x, bot_y, self.PIPE_W, self.GROUND_Y - bot_y),
                2,
            )

        # Sol
        pygame.draw.rect(
            self._surface,
            (222, 216, 149),
            (0, self.GROUND_Y, self.SCREEN_W, self.SCREEN_H - self.GROUND_Y),
        )
        pygame.draw.line(
            self._surface, (84, 56, 71), (0, self.GROUND_Y), (self.SCREEN_W, self.GROUND_Y), 3
        )

        # Oiseau
        bird_rect = pygame.Rect(
            int(self.BIRD_X), int(self.bird_y), self.BIRD_W, self.BIRD_H
        )
        pygame.draw.ellipse(self._surface, (248, 216, 48), bird_rect)
        pygame.draw.ellipse(self._surface, (84, 56, 71), bird_rect, 2)
        # Œil
        pygame.draw.circle(
            self._surface,
            (255, 255, 255),
            (int(self.BIRD_X + self.BIRD_W * 0.72), int(self.bird_y + self.BIRD_H * 0.35)),
            5,
        )
        pygame.draw.circle(
            self._surface,
            (0, 0, 0),
            (int(self.BIRD_X + self.BIRD_W * 0.76), int(self.bird_y + self.BIRD_H * 0.35)),
            2,
        )

        # Score
        score_surf = self._font.render(f"Score: {self.score}", True, (255, 255, 255))
        self._surface.blit(score_surf, (self.SCREEN_W // 2 - score_surf.get_width() // 2, 25))

        if self.render_mode == "human":
            pygame.event.pump()
            self._display.blit(self._surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self._surface), axes=(1, 0, 2))

    def close(self):
        if self._display is not None:
            pygame.display.quit()
            pygame.quit()
            self._display = None
            self._surface = None


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    print("Validation Gymnasium (`check_env`) sur CustomFlappyBirdEnv...")
    custom_env = CustomFlappyBirdEnv(render_mode=None)
    check_env(custom_env)
    print("[OK] CustomFlappyBirdEnv respecte 100% du standard Gymnasium !")
    custom_env.close()
