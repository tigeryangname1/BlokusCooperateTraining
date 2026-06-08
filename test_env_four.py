# test_env_four.py
from blokus_env_four_color import BlokusFourColorEnv

def main():
    env = BlokusFourColorEnv(
        min_obstacles=0,
        max_obstacles=0,
    )
    obs, info = env.reset()
    env.render()

    done = False
    step = 0
    while not done and step < 1110:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"step={step}, reward={reward}, info={info}")
        step += 1

    env.render()

if __name__ == "__main__":
    main()
