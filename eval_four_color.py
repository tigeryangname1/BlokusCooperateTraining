from stable_baselines3 import PPO
from blokus_env_four_color import BlokusFourColorEnv
from blokus import BLUE, YELLOW, RED, GREEN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np

def run_random_episode(env, max_steps=600, render=True):
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    last_info = {}

    while not done and step < max_steps:
        # 1. 取得最新解包的原始環境，用以獲取當前合法的 current_mask
        raw_env = env.envs[0] if hasattr(env, "envs") else env
        current_mask = raw_env.current_mask  # 這是一個長度與 action_space 相同的 bool 陣列或列表
        
        # 2. 轉換為符合 Gymnasium 規範的 np.ndarray (dtype=np.int8)
        #    Gymnasium 的 sample(mask=...) 內部預期遮罩格式為 1 代表合法，0 代表無效
        sample_mask = np.array(current_mask, dtype=np.int8)

        # 3. 檢查是否還有任何合法動作（防止所有 mask 皆為 False 導致 sample 報錯）
        if np.any(sample_mask == 1):
            # 核心修正：只在 mask=1 的合法 Action ID 中隨機挑選一個
            # 注意：這裡如果是經過含有 VecEnv 包裹的環境，需要用單個環境的 action_space
            action = raw_env.action_space.sample(mask=sample_mask)
            
            # 如果你的環境被外層包裹（如 VecEnv），傳給 env.step() 的動作通常需要包成陣列或列表
            # 如果直接使用 raw 動作噴錯，請改用： action = [action] 或 np.array([action])
            if hasattr(env, "envs"):
                action = np.array([action])
        else:
            # 如果完全沒有合法動作可以選（例如全部被遮罩），通常代表該玩家必須 pass 或遊戲已死局
            # 這裡可以根據你環境的設計給予一個預設的 pass 動作（例如動作空間最後一個 ID）
            # 或者由環境在下一行 env.step 時自行處理。這裡暫定給予 0 或維持原樣
            action = env.action_space.sample()

        # 4. 執行安全的合法隨機動作
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
    # ==================== 關鍵修正區域 ====================
    # 2. 獲取單個非向量化環境（如果是用 DummyVecEnv 包裹，需要用 env.envs[0] 取得原始環境）
    #    若 env 本身就是原始自訂環境，可以直接寫 raw_env = env
    raw_env = env.envs[0] if hasattr(env, "envs") else env
    raw_env = env
    # 3. 根據放完棋子後的新狀態，重新計算合法步與遮罩
    current_color = raw_env.colors[raw_env.current_color_index]
    legal_moves = raw_env.state.generate_legal_moves(current_color)
    
    # 4. 重新為環境更新快取（這樣等一下 model.predict 內部呼叫 action_masks() 時才不會錯位）
    raw_env.current_padded_moves, raw_env.current_mask = raw_env._get_padded_moves_and_mask(legal_moves)
    
    # 5. 將最新、對齊後的遮罩丟進 _get_obs()
    obs = raw_env._get_obs(raw_env.current_mask)
    # =====================================================
    step = 1
    
    done = False
    #step = 0
    total_reward = 0
    last_info = {}
    random_prefix_steps = 1
    while not done and step < max_steps:
        # 1. 取得最新解包的原始環境
        raw_env = env.envs[0] if hasattr(env, "envs") else env
        
        # 2. 【核心修正】明確把目前最新計算出來的 current_mask 丟給模型
        #    這會強制 SB3 在進行機率分佈預測時，將 current_mask 為 False 的地方壓成 0
        action, _ = model.predict(
            obs, 
            deterministic=deterministic, 
            action_masks=raw_env.current_mask  # 這裡強制指定
        )

        # 3. 執行動作。在 step 內部結尾處，你應該已經透過快取機制，自動把 self.current_mask 更新為下一手玩家的了
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        last_info = info

        # 4. 如果遊戲還沒結束，保險起見，我們同步把 obs 的欄位和環境快取做對齊
        if not done:
            # 確保環境再次解包（若被多層包裹）
            raw_env = env.envs[0] if hasattr(env, "envs") else env
            
            # 如果你的 env.step() 內部沒有主動更新下一個人的快取，才需要解開下面這兩行：
            # current_color = raw_env.colors[raw_env.current_color_index]
            # raw_env.current_padded_moves, raw_env.current_mask = raw_env._get_padded_moves_and_mask(raw_env.state.generate_legal_moves(current_color))
            
            # 重新打包 obs，確保 obs 字典和下一輪要用的遮罩完美同步
            obs = raw_env._get_obs(raw_env.current_mask)

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
    env = BlokusFourColorEnv()

    model = MaskablePPO.load("ppo_blokus_four_color_soft", env=env)

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
