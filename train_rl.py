from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from blokus_env import BlokusSingleColorEnv

env = BlokusSingleColorEnv()
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

model.save("ppo_blokus_singlecolor_softsuccess")
