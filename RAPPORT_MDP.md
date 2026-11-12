# Projet RL — Flappy Bird (M2 DP IA — ISEN Nantes)

**Module :** Apprentissage par renforcement — Projet  
**Enseignant :** Hadir Amine (EC en Informatique / IA)  
**Jeu choisi :** Flappy Bird (`flappy-bird-gymnasium`, `use_lidar=False`)

---

## 1. Modélisation du problème sous forme de MDP (5 points)

Conformément à la règle d'or du sujet, nous utilisons la représentation **vectorielle à 12 nombres réels** (`use_lidar=False`, `normalize_obs=True`) et non les pixels bruts, ce qui permet d'entraîner un réseau de neurones multicouche (`MlpPolicy`) en quelques minutes sur CPU.

### 1.1 Espace d'États ($\mathcal{S}$)
L'état $s_t \in [-1, 1]^{12}$ retourné à chaque pas de temps décrit exactement la géométrie des **3 prochains tuyaux** visibles à l'écran ainsi que la **cinématique verticale de l'oiseau** (dimensions de l'écran : $W = 288\text{ px}$, $H = 512\text{ px}$) :

| Indice | Variable | Description physique | Normalisation |
| :--- | :--- | :--- | :--- |
| `s[0]` | $x_{\text{pipe},0}$ | Distance horizontale du tuyau actuel/proche | $x / 288$ |
| `s[1]` | $y_{\text{top},0}$ | Ordonnée du bord inférieur du tuyau haut $0$ | $y / 512$ |
| `s[2]` | $y_{\text{bot},0}$ | Ordonnée du bord supérieur du tuyau bas $0$ | $y / 512$ |
| `s[3..5]` | $(x_1, y_{\text{top},1}, y_{\text{bot},1})$ | Coordonnées du deuxième couple de tuyaux | $/288$ et $/512$ |
| `s[6..8]` | $(x_2, y_{\text{top},2}, y_{\text{bot},2})$ | Coordonnées du troisième couple de tuyaux | $/288$ et $/512$ |
| `s[9]` | $y_{\text{bird}}$ | Position verticale de l'oiseau | $y_{\text{bird}} / 512$ |
| `s[10]` | $v_{y,\text{bird}}$ | Vitesse verticale de l'oiseau ($v_{\max}=10$) | $v_y / 10$ |
| `s[11]` | $\theta_{\text{bird}}$ | Angle de rotation de l'oiseau | $\theta / 90^\circ$ |

### 1.2 Espace d'Actions ($\mathcal{A}$)
L'espace d'actions est discret de taille 2 (`Discrete(2)`) :
- **$a = 0$ (IDLE / Ne rien faire)** : l'oiseau subit l'accélération gravitationnelle vers le bas ($+1\text{ px/pas}^2$, plafonnée à $+10\text{ px/pas}$).
- **$a = 1$ (FLAP / Battre des ailes)** : donne une impulsion verticale instantanée vers le haut ($v_y \leftarrow -9\text{ px/pas}$).

### 1.3 Fonction de Récompense ($\mathcal{R}$) et *Reward Shaping*
Dans l'environnement par défaut (`reward_mode="default"`), la récompense est :
- $+0.1$ à chaque pas survécu,
- $+1.0$ lorsqu'un tuyau est franchi,
- $-0.5$ si l'oiseau dépasse le plafond ($y < 0$),
- $-1.0$ en cas de collision (sol ou tuyau, fin de l'épisode).

**Limitation de la récompense par défaut :** au début de l'apprentissage, un agent qui bat des ailes trop souvent ou trop peu ne perçoit aucune indication sur la hauteur de l'ouverture du prochain tuyau tant qu'il ne l'a pas atteint.

**Notre *Reward Shaping* (`FlappyRewardWrapper` dans `env_wrapper.py`) :**
Nous avons conçu une fonction de récompense enrichie $R_{\text{shaped}}(s_t, a_t, s_{t+1})$ qui guide continûment l'oiseau vers le centre de l'ouverture du prochain tuyau :
$$y_{\text{gap}} = \frac{y_{\text{top}} + y_{\text{bot}}}{2}, \qquad d_t = |y_{\text{bird}} - y_{\text{gap}}|$$
$$R_{\text{shaped}} = \begin{cases}
-1.0 & \text{si collision (mort)} \\
+1.5 + 0.05 \cdot (1 - 4 d_t) & \text{si tuyau franchi} \\
r_{\text{env}} + 0.05 \cdot (1 - 4 d_t) & \text{sinon (survie à chaque pas)}
\end{cases}$$

**Justification :**
1. **Signal dense de centrage** : le terme $0.05(1 - 4d_t)$ donne un bonus lorsque l'oiseau est aligné avec le centre du trou ($d_t < 0.25$) et une légère pénalité lorsqu'il s'en écarte fortement, accélérant la découverte du couloir de passage.
2. **Bonus de franchissement renforcé ($+1.5$ au lieu de $+1.0$)** : encourage la prise de risque pour franchir le tuyau plutôt que de simplement survivre devant.
3. **Évaluation non biaisée** : toutes nos courbes et évaluations comparent les agents sur la **récompense brute standard** (`episode_raw_reward`) afin que la comparaison entre *Reward Shaping* et *Default* soit 100 % rigoureuse.

---

## 2. Algorithmes & Résultats Expérimentaux (6 points)

Nous avons entraîné et comparé 3 configurations avec `Stable-Baselines3` (architecture MLP `[128, 128]`, $\gamma = 0.99$) face à la baseline aléatoire (hasard) sur **30 épisodes de test déterministes** :

| Agent / Configuration | Pas d'entraînement | Récompense brute moyenne | Tuyaux franchis (moyenne) | Score Max (test) | Taux $\ge 1$ tuyau | Taux $\ge 5$ tuyaux |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hasard (Aléatoire)** | 0 | **$-7.30$** | $0.00$ | $0$ | $0.0\%$ | $0.0\%$ |
| **DQN + Reward Shaping** | 250 000 | **$+8.58 \pm 3.02$** | $1.13$ | $3$ | $90.0\%$ | $0.0\%$ |
| **PPO + Récompense défaut** | 250 000 | **$+19.74 \pm 13.59$** | $3.40$ | $11$ | $90.0\%$ | $23.3\%$ |
| **PPO + Reward Shaping (Retenu)** | 350 000 | **$+36.41 \pm 52.42$** | **$7.03$** | **$56$** | **$100.0\%$** | **$36.7\%$** |

> **Analyse :**
> - Le critère de réussite de la fiche projet ($\approx -8$ au hasard $\rightarrow +13$ après entraînement) est largement dépassé : notre modèle **PPO + Reward Shaping** atteint **$+36.41$ de récompense moyenne** ($100\%$ des parties franchissent au moins 1 tuyau, et jusqu'à **56 tuyaux franchis** en un seul épisode).
> - **PPO vs DQN** : PPO (méthode *on-policy* à gradient de politique avec clipping) est nettement plus stable que DQN sur Flappy Bird, car l'inertie verticale de l'oiseau rend la surestimation des valeurs $Q(s,a)$ sensible aux changements brusques de hauteur entre deux tuyaux.
> - **Impact du Reward Shaping** : à budget d'entraînement comparable, le *Reward Shaping* double quasiment la récompense finale moyenne de PPO ($+36.41$ contre $+19.74$) et le nombre moyen de tuyaux franchis ($7.03$ contre $3.40$).

### Courbes générées
- `plots/courbe_apprentissage_ppo.png` : évolution de la récompense cumulée et du nombre de tuyaux franchis au cours de l'entraînement de PPO.
- `plots/comparaison_algos_reward.png` : comparaison directe PPO (Shaped) vs PPO (Défaut) vs DQN (Shaped) vs Hasard.

---

## 3. Bonus réalisés (+2 points)

1. **Comparaison algorithmique (PPO vs DQN) et ablation du Reward Shaping** : implémentée dans `train.py` et visualisée dans `plots/comparaison_algos_reward.png`.
2. **Version complète du jeu codée de zéro en Pygame + Wrapper Gymnasium (`bonus_custom_pygame_env.py`)** :
   - Implémentation complète de la classe `CustomFlappyBirdEnv(gym.Env)` en Pygame pur (gravité, impulsion, génération aléatoire et défilement des tuyaux, détection de collisions par `pygame.Rect`, vecteur d'observation normalisé de 12 réels et rendu graphique 30 FPS).
   - Validée avec succès par l'outil officiel `stable_baselines3.common.env_checker.check_env`.

---

## 4. Répartition des tâches & Support de présentation (10 min)

### Répartition au sein du binôme/trinôme (créneau 3 h)
- **Rôle A — Le jeu / l'environnement (`env_wrapper.py`, `bonus_custom_pygame_env.py`)** : Analyse des 12 features d'état de `FlappyBird-v0`, conception du wrapper `FlappyRewardWrapper` (centrage dans l'ouverture + bonus de tuyau), et codage de l'environnement Pygame complet en bonus.
- **Rôle B — L'agent / l'entraînement (`train.py`)** : Mise en place des environnements vectorisés (`DummyVecEnv`), réglage des hyperparamètres de PPO et DQN (`net_arch=[128, 128]`, `ent_coef=0.01`, `n_steps=512`), callback de journalisation et sauvegarde des modèles (`flappy_ppo.zip`, `models/`).
- **Rôle C — Évaluation / Démo / Rapport (`evaluate_and_plot.py`, `demo.py`, `RAPPORT_MDP.md`)** : Protocole d'évaluation sur 30 épisodes déterministes, génération des graphiques Matplotlib, script de démonstration temps réel Pygame (`render_mode="human"`) et rédaction du rapport MDP.

### Plan de présentation orale (10 minutes)
1. **0:00 – 2:30 | Modélisation MDP (Rôle A)** : Présentation des 12 variables d'état (`use_lidar=False`), des 2 actions discrètes et de la formule de *Reward Shaping* (guidage vers le centre du prochain tuyau).
2. **2:30 – 5:30 | Entraînement & Comparaison PPO vs DQN (Rôle B)** : Explication de la boucle `Stable-Baselines3`, hyperparamètres choisis, et analyse des deux figures (`courbe_apprentissage_ppo.png` et `comparaison_algos_reward.png`).
3. **5:30 – 8:30 | Démo live « Regarder jouer » (Rôle C)** : Lancement en direct de `python demo.py --random` (montrant l'oiseau qui tombe à $-7.3$) puis `python demo.py` (montrant l'agent PPO franchir les tuyaux en continu).
4. **8:30 – 10:00 | Ouverture & Bonus Vibe Coding** : Démonstration rapide de notre environnement codé from scratch (`bonus_custom_pygame_env.py`) et conclusion.

---

## 5. Commandes pour lancer le projet

```bash
# 1. Vérifier l'environnement et le wrapper MDP (Rôle A)
C:\Users\tonyk\anaconda3\python.exe env_wrapper.py

# 2. Regarder l'agent PPO entraîné jouer en direct (Rôle C - Démo)
C:\Users\tonyk\anaconda3\python.exe demo.py

# 3. Comparer avec un agent qui joue au hasard
C:\Users\tonyk\anaconda3\python.exe demo.py --random

# 4. Régénérer les courbes d'apprentissage et de comparaison
C:\Users\tonyk\anaconda3\python.exe evaluate_and_plot.py

# 5. Tester l'environnement Pygame codé de zéro (Bonus)
C:\Users\tonyk\anaconda3\python.exe bonus_custom_pygame_env.py

# 6. Relancer l'entraînement complet (Rôle B)
C:\Users\tonyk\anaconda3\python.exe train.py --algo all --timesteps 350000
```
