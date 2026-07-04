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
class BlokusCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 處理棋盤相關的空間特徵
        # board(1) + attachment_maps(4) + extension_maps(4) = 9 channels
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 計算 CNN 輸出維度
        cnn_output_dim = 64 * 20 * 20  # 25600
        
        # 處理非空間特徵（remaining、turn）
        # 21*4 + 1 = 85
        other_dim = 85
        
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + other_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        board = observations["board"].unsqueeze(1).float()  # (B,1,20,20)
        attach = observations["attachment_maps"].float()     # (B,4,20,20)
        ext = observations["extension_maps"].float()         # (B,4,20,20)
        
        spatial = torch.cat([board, attach, ext], dim=1)    # (B,9,20,20)
        cnn_out = self.cnn(spatial)
        
        remaining = torch.cat([
            observations["remaining_blue"].float(),
            observations["remaining_yellow"].float(),
            observations["remaining_red"].float(),
            observations["remaining_green"].float(),
            observations["turn"].float(),
        ], dim=1)
        
        combined = torch.cat([cnn_out, remaining], dim=1)
        return self.mlp(combined)

def make_env():
    def _init():
        return BlokusFourColorEnv() # 替換成你的環境類別
    return _init

def main():
    # 假設你的電腦是 8 核心，開啟 4~6 個並行進程
    num_cpu = 6 
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    # 定義更適合棋盤與複雜觀測空間的神經網路架構
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # 策略網路 (Policy Network) 稍微加深
            vf=[256, 256, 128]   # 價值網路 (Value Network) 稍微加深
        )
    )

    # policy_kwargs = dict(
    #     features_extractor_class=BlokusCNNExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch=dict(
    #         pi=[128, 128],
    #         vf=[128, 128],
    #     )
    # )
    '''
    model = MaskablePPO.load("ppo_blokus_four_color_soft", env=env)
    model.learn(total_timesteps=400_000, reset_num_timesteps=False)
    '''
    # 1. 建立支援 Mask 的 PPO 模型
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=0.003, # 測試訓練改0.003 大概要跑300萬步 
        # learning_rate=1e-3,     # 保持 3e-4，或根據收斂速度微調至 1e-4 # 3e-4覺得太慢 加快一下學習率
        n_steps=4096,           # 每輪收集的樣本數
        batch_size=512,         # 【強烈建議修改】放大 batch size 穩定梯度、加速 GPU 運算
        n_epochs=8,            # 每次採樣後重複優化的次數
        gamma=0.99,             # 重視長期回報（非常契合 Blokus 終局結算）
        gae_lambda=0.95,        # 保持穩定的優勢函數估計
        ent_coef=0.05,          # 【重要】強迫 Agent 探索多元的落子位置與卡位策略，避免過早死鎖 
        vf_coef=0.5,            # 價值網路損失權重（配合調整後的 Reward 尺度）
        max_grad_norm=0.5,      # 梯度裁剪，防止權重更新過大導致策略崩塌
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/" # 建議開啟 TensorBoard 觀察回報與 explained_variance
    )
    # 開啟語法: tensorboard_logs --logdir
    # 提高 entropy 鼓勵在死局前找其他出路
    # learning_rate=1e-4 調小學習率，走穩一點

    # 2. 開始訓練（MaskablePPO會自動去環境的 obs 裡找 "action_mask" 並套用）
    model.learn(total_timesteps=1000000)

    #'''
    model.save("ppo_blokus_four_color_soft")

if __name__ == "__main__":
    main()
