from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from blokus_env_four_color import BlokusFourColorEnv

def main():
    env = BlokusFourColorEnv(
        min_obstacles=0,
        max_obstacles=0,
    )

    check_env(env, warn=True)
    '''
    model = PPO.load("ppo_blokus_four_color_soft", env=env)
    model.learn(total_timesteps=2000_000, reset_num_timesteps=False)
    '''
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    model.learn(total_timesteps=400_000)
    #'''
    model.save("ppo_blokus_four_color_soft")

if __name__ == "__main__":
    main()
