# test_env_four.py
from blokus_env_four_color import BlokusFourColorEnv
import numpy as np
    # ===== 偵錯測試區塊：驗證 mask 正確性 =====
    # 建議放在 _get_padded_moves_and_mask 改完之後，訓練前先跑幾步確認



def main():
    env = BlokusFourColorEnv()
    env.debug_mask_validation(n_steps=10)

if __name__ == "__main__":
    main()
