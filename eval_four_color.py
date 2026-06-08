from stable_baselines3 import PPO
from blokus_env_four_color import BlokusFourColorEnv
from blokus import BLUE, YELLOW, RED, GREEN

def run_random_episode(env, max_steps=600, render=True):
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    last_info = {}

    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        last_info = info

    reason = last_info.get("reason", "unknown")
    left_total = last_info.get("left_total", None)

    if render:
        env.render()
    return {
        "steps": step,
        "total_reward": total_reward,
        "reason": reason,
        "left_total": left_total,
    }
def apply_random_big_move(env, color=None):
    # 直接從 env.state 產生合法步
    state = env.state
    if state is None:
        return

    # 如果 color=None，就用 current_color_index 對應那一色
    if color is None:
        color = env.colors[env.current_color_index]

    legal = state.generate_legal_moves(color)
    if not legal:
        return

    # 挑出最大尺寸的幾種棋，假設 top_k = 5
    top_k = 5
    sorted_moves = sorted(legal, key=lambda m: len(m["shape"]), reverse=True)
    candidates = sorted_moves[:top_k] if len(sorted_moves) > top_k else sorted_moves

    import random
    move = random.choice(candidates)

    # 手動 apply
    new_state = state.apply_move(color, move["shape"], move["x"], move["y"], move["piece"])
    env.state = new_state

def run_one_episode(env, model, max_steps=600, render=True, deterministic=False):
    obs, info = env.reset()
    # 先手動放一顆「大塊隨機」的
    apply_random_big_move(env)
    # 放完後重新抓 obs
    obs = env._get_obs()
    step = 1
    
    done = False
    #step = 0
    total_reward = 0
    last_info = {}
    random_prefix_steps = 1
    while not done and step < max_steps:

        action, _ = model.predict(obs, deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        last_info = info
        
    if render:
        env.render()

    # 從最後一步的 info 把結果拿出來（可能是空 dict，就保險處理一下）
    reason = last_info.get("reason", "unknown")
    left_total = last_info.get("left_total", None)
    left_blue = last_info.get("left_blue", None)
    left_yellow = last_info.get("left_yellow", None)
    left_red = last_info.get("left_red", None)
    left_green = last_info.get("left_green", None)
    empty_cells = last_info.get("empty_cells", None)

    return {
        "steps": step,
        "total_reward": total_reward,
        "reason": reason,
        "left_total": left_total,
        "left_each": {
            "B": left_blue,
            "Y": left_yellow,
            "R": left_red,
            "G": left_green,
        },
        "empty_cells": empty_cells,
    }
    
def run_one_episode_new(env, model, render=False, deterministic=False):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    invalid_steps = 0
    last_info = {}

    while not (done or truncated):

        # 先用 deterministic=False 測試
        action, _ = model.predict(obs, deterministic)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if info.get("invalid_action", False):
            invalid_steps += 1

        last_info = info

    if render:
        env.render()
    # reason / left_total 處理（含 max_steps timeout）
    if "reason" in last_info:
        reason = last_info["reason"]
    elif truncated:
        reason = "max_steps"
    else:
        reason = "unknown"

    if "left_total" in last_info:
        left_total = last_info["left_total"]
        left_each = {
            "B": last_info["left_blue"],
            "Y": last_info["left_yellow"],
            "R": last_info["left_red"],
            "G": last_info["left_green"],
        }
    else:
        left_total = None
        left_each = {"B": None, "Y": None, "R": None, "G": None}

    return {
        "steps": steps,
        "total_reward": total_reward,
        "reason": reason,
        "left_total": left_total,
        "left_each": left_each,
        "invalid_steps": invalid_steps,
    }

def main():
    # ⚠️ 這裡的障礙設定請跟訓練時保持一致
    env = BlokusFourColorEnv(
        min_obstacles=0,
        max_obstacles=0,
    )

    model = PPO.load("ppo_blokus_four_color_soft", env=env)

    print("=== 四色 PPO 模型評估 ===\n")

    results = []
    num_episodes = 2

    for i in range(num_episodes):
        print(f"\n=== Episode {i} ===")
        result = run_one_episode(env, model, render=True,deterministic=False)
        results.append(result)

        print(
            f"steps={result['steps']}, "
            f"reward={result['total_reward']:.1f}, "
            f"reason={result['reason']}"
        )
        print(
            "left_total =", result["left_total"],
            "| B=", result["left_each"]["B"],
            "Y=", result["left_each"]["Y"],
            "R=", result["left_each"]["R"],
            "G=", result["left_each"]["G"],
            "empty_cells=", result["empty_cells"],
        )
        print("-" * 40)

    # 簡單統計
    valid_results = [r for r in results if r["left_total"] is not None]
    if valid_results:
        avg_steps = sum(r["steps"] for r in valid_results) / len(valid_results)
        avg_reward = sum(r["total_reward"] for r in valid_results) / len(valid_results)
        avg_left_total = sum(r["left_total"] for r in valid_results) / len(valid_results)

        print("\n=== 平均結果 ===")
        print(f"平均步數        = {avg_steps:.1f}")
        print(f"平均 total_reward = {avg_reward:.1f}")
        print(f"平均 left_total = {avg_left_total:.1f}")
    print("\n=== 隨機策略對照 ===")
    for i in range(1):
        r = run_random_episode(env)
        print(
            f"Random {i}: steps={r['steps']}, "
            f"reward={r['total_reward']:.1f}, "
            f"reason={r['reason']}, "
            f"left_total={r['left_total']}"
        )


if __name__ == "__main__":
    main()
