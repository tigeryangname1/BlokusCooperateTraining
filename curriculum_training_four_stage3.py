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
def make_env():
    def _init():
        return BlokusFourColorEnv(max_curriculum_steps=9999)  # 完整遊戲
    return _init

def main():
    num_cpu = 6
    
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])
    
    # 載入階段二的權重
    model = MaskablePPO.load(
        "ppo_blokus_stage2",
        env=env,
        custom_objects={
            "learning_rate": 0.00003,  # 再降低 lr
            "ent_coef": 0.01,          # 收斂階段少探索
        }
    )
    
    print("=== 階段三：完整遊戲 ===")
    model.learn(
        total_timesteps=3_000_000,
        reset_num_timesteps=False,  # 接續階段二的 timesteps
    )
    
    model.save("ppo_blokus_stage3")
    print("階段三完成，模型儲存為 ppo_blokus_stage3")
    env.close()

if __name__ == "__main__":
    main()