from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from blokus_env_four_color import BlokusFourColorEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import torch.nn as nn
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 關閉 oneDNN 裝了tensorflow後一直有錯誤訊息，有點煩這邊先關掉
# training_stage2.py
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from blokus_env_four_color import BlokusFourColorEnv

def make_env(curriculum_steps):
    def _init():
        return BlokusFourColorEnv(max_curriculum_steps=curriculum_steps)
    return _init

def main():
    num_cpu = 6
    curr_steps = 60 
    
    env = SubprocVecEnv([make_env(curr_steps) for _ in range(num_cpu)])
    
    # 載入階段一的權重
    model = MaskablePPO.load(
        "ppo_blokus_stage1",
        env=env,
        # 保留原本的超參數，只覆蓋需要調整的
        custom_objects={
            "learning_rate": 0.0001,  # 降低 lr，細緻調整
            "ent_coef": 0.03,         # 稍微降低探索
        }
    )
    
    print("=== 階段二：訓練前 60 步 ===")
    model.learn(
        total_timesteps=1_000_000,
        reset_num_timesteps=False,  # 接續階段一的 timesteps
    )
    
    model.save("ppo_blokus_stage2")
    print("階段二完成，模型儲存為 ppo_blokus_stage2")
    env.close()

if __name__ == "__main__":
    main()