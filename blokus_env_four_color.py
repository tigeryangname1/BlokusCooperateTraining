import numpy as np
import gymnasium as gym
import math

from gymnasium import spaces
from blokus import GameState, BOARD_SIZE, BLUE, YELLOW, RED, GREEN, BASE_PIECES, ALL_PIECES


class BlokusFourColorEnv(gym.Env):
    """
    四色合作 Blokus 環境：

    - 控制 BLUE / YELLOW / RED / GREEN 四個顏色
    - 四色輪流下子，同一個 PPO policy
    - reward:
        - 每一步：放下的格子數（不分顏色）
        - 終局：根據四色剩餘格數總和做軟成功獎勵
    - 終局條件：四個顏色都沒有合法步數
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        max_candidates: int = 36400,
    ):
        super().__init__()
        # 請把這段加到你的環境 __init__ 裡面
        self.time_stats = {
            "block1_init_legal": 0.0,
            "block2_timeout_check": 0.0,
            "block3_all_color_check": 0.0,
            "block4_ensure_color": 0.0,
            "block5_get_move": 0.0,
            "block6_corner_before": 0.0,
            "block7_apply_move": 0.0,
            "block8_corner_after_calc": 0.0,
            "block9_final_check_switch": 0.0,
    
            # 區塊 9 細分計時器
            "b9_1_a_any_legal_moves": 0.0,    # 核心：檢查四色是否都沒步 (self._any_legal_moves_for_any_color)
            "b9_1_b_compute_left_total": 0.0, # 終局：計算剩餘方塊總數
            "b9_1_c_compute_left_each": 0.0,  # 終局：計算各色剩餘方塊
            "b9_1_d_compute_empty_cell": 0.0, # 終局：計算棋盤空格數
            
            "b9_2_switch_color": 0.0,         
            "b9_3_gen_next_legal_moves": 0.0, 
            "b9_4_get_padded_and_mask": 0.0,  
            "b9_5_get_obs_and_misc": 0.0,
        }
        self.step_call_count = 0  # 記錄 step 被呼叫了幾次

        # 四個顏色都由同一個 agent 控制
        self.colors = [BLUE, YELLOW, RED, GREEN] # state 裡面也有 colors ，為了判斷 legal_move 的 color_has_moves 變數暫存用，但這邊是為了顏色輪轉讓 agent 知道下一個是誰
        self.current_color_index = 0  # 0=BLUE, 1=YELLOW, 2=RED, 3=GREEN
        self.max_candidates = max_candidates

        # 棋子順序（四色共用同一組）
        self.piece_names = sorted(BASE_PIECES.keys())
        self.num_pieces = len(self.piece_names)

        # === 修正後的 Observation Space ===
        # 直接把 action_mask 放進同一個 Dict 裡面，與原本的特徵並存
        self.observation_space = spaces.Dict(
            {
                # 動作遮罩：1 代表合法，0 代表被遮罩（無效）
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_candidates,),
                    dtype=np.int8,
                ),
                # 棋盤狀態
                "board": spaces.Box(
                    low=0,
                    high=5,
                    shape=(BOARD_SIZE, BOARD_SIZE),
                    dtype=np.int8,
                ),
                # 各色手牌剩餘狀態
                "remaining_blue": spaces.Box(
                    low=0, high=1, shape=(self.num_pieces,), dtype=np.int8
                ),
                "remaining_yellow": spaces.Box(
                    low=0, high=1, shape=(self.num_pieces,), dtype=np.int8
                ),
                "remaining_red": spaces.Box(
                    low=0, high=1, shape=(self.num_pieces,), dtype=np.int8
                ),
                "remaining_green": spaces.Box(
                    low=0, high=1, shape=(self.num_pieces,), dtype=np.int8
                ),
                # 當前回合提示
                "turn": spaces.Box(
                    low=0, high=3, shape=(1,), dtype=np.int8
                ),
            }
        )

        # === Action Space ===
        self.action_space = spaces.Discrete(self.max_candidates)

        # 內部狀態
        self.state: GameState | None = None
        self._last_legal_moves: list[dict] = []
        self.current_steps = 0
        self.max_steps = 600  # 安全上限

        # 1. 定義 21 個棋子的固定順序（從已展開的 ALL_PIECES 取得 21 個棋子名稱並排序）
        self.ALL_PIECE_NAMES = sorted(list(ALL_PIECES.keys())) 
        
        # 2. 預先建立全域固定的動作空間 (All Possible Candidates)
        self.GLOBAL_CANDIDATE_MOVES = []
        
        # 建立快速查詢字典：輸入 (piece, o_idx, x, y) 查 Action ID
        self.MOVE_TO_ACTION_ID = {}
        
        action_id = 0
        for piece_name in self.ALL_PIECE_NAMES:
            # 這裡正確取得該棋子「已經展開的所有方向列表」
            # 例如：1x1 方塊長度為 1；L型方塊長度為 8
            orientations = ALL_PIECES[piece_name] 
            
            for o_idx, shape in enumerate(orientations):
                for x in range(20):      # 20x20 棋盤
                    for y in range(20):
                        move_obj = {
                            "piece": piece_name,
                            "shape": shape,
                            "x": x,
                            "y": y,
                            "o_idx": o_idx 
                        }
                        self.GLOBAL_CANDIDATE_MOVES.append(move_obj)
                        
                        # 記錄唯一的對應關係
                        self.MOVE_TO_ACTION_ID[(piece_name, o_idx, x, y)] = action_id
                        action_id += 1
                        
        # 這是你模型【精準且絕對固定】的 Action Space 總大小
        self.total_action_space_size = len(self.GLOBAL_CANDIDATE_MOVES)
        
        # print(f"=== 動作空間初始化完成！精準總動作數: {self.total_action_space_size} ===")
        
        # PPO 每次只需要讀取這個固定的 Candidate 指標，不需要重複建立
        self.current_padded_moves = self.GLOBAL_CANDIDATE_MOVES


    # -----------------------
    # Gym API
    # -----------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = GameState()
        self._last_legal_moves = [] 
        self.current_steps = 0
        self.current_color_index = 0  # 從 BLUE 開始

        # 確保第一個顏色有步可走（雖然開局第一步通常都有，但建議保持邏輯一致）
        self._ensure_current_color_has_moves()
        
        # 取得當前（BLUE）的第一步所有合法動作
        current_color = self.colors[self.current_color_index]
        legal_moves = self.state.generate_legal_moves(current_color)
        self._last_legal_moves = legal_moves  # 紀錄下來防呆或紀錄
        
        # 快取當前的對齊動作與遮罩
        self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(legal_moves)
        
        obs = self._get_obs(self.current_mask)

        info = {}
        return obs, info
    def _compute_empty_board_cells(self) -> int:
        board = self.state.board # 假設 board 是 2D array，空格為 0
        empty = 0
        for row in board:
            for cell in row:
                if cell == 0:
                    empty += 1
        return empty
    def action_masks(self) -> list[bool]:
        # 核心改動：直接回傳快取住的 mask，絕對不會因為呼叫時間點不同而錯位！
        return self.current_mask
    
    def _write_time_stats(self):
        """將累積時間覆寫到檔案中"""
        with open("step_time_report.txt", "w", encoding="utf-8") as f:
            f.write(f"=== Step 效能累積報告 (總呼叫次數: {self.step_call_count}) ===\n")
            f.write(f"{'區塊名稱':<30}{'累積總耗時 (秒)':<20}{'單次平均 (毫秒)':<20}\n")
            f.write("-" * 70 + "\n")
            for block, total_time in self.time_stats.items():
                avg_ms = (total_time / self.step_call_count * 1000) if self.step_call_count > 0 else 0
                f.write(f"{block:<30}{total_time:<20.4f}{avg_ms:<20.4f}\n")

    def step(self, action: int):
        import time
        self.step_call_count += 1
        
        # --- [區塊 1: 基礎初始化與合法步生成] ---
        t_start = time.perf_counter()
        assert self.state is not None, "請先呼叫 reset()"
        self.current_steps += 1
        self.state
        current_color = self.colors[self.current_color_index]
        
        # --- [區塊 3: 四色皆無步終局檢查] ---
        t_start = time.perf_counter()
        if not self._any_legal_moves_for_any_color():
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = self._final_reward(left_total, left_each)
            empty_cells = self._compute_empty_board_cells()
            self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask([])
            obs = self._get_obs(self.current_mask)
            terminated = True
            truncated = False
            info = {
                "reason": "no_legal_moves_all_1",
                "left_total": left_total,
                "left_blue": left_each[BLUE],
                "left_yellow": left_each[YELLOW],
                "left_red": left_each[RED],
                "left_green": left_each[GREEN],
                "empty_cells": empty_cells,
            }
            self.time_stats["block3_all_color_check"] += (time.perf_counter() - t_start)
            self._write_time_stats()  # 覆寫輸出檔案
            return obs, final_reward, terminated, truncated, info
        self.time_stats["block3_all_color_check"] += (time.perf_counter() - t_start)
        # 【關鍵優化點】: 不再一開始就 generate_legal_moves！
        # 直接拿我們之前做好的生死快取來判斷：
        if not self.state.color_has_moves[current_color]:
            # 如果這顏色沒步可走，直接執行你提供的「換下一色、打包 Mask、常規返回」流程
            self._switch_color()
            next_color = self.colors[self.current_color_index]
            
            # 為下一個顏色生成合法步（因為他是下一個要正式下棋的人，這裡才必須拿全量 moves 去做 Mask）
            next_legal_moves = self.state.generate_legal_moves(next_color)
            self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(next_legal_moves)
            obs = self._get_obs(self.current_mask)
            
            reward = 0.0  # 沒步可走的玩家，這一步的 reward 為 0
            terminated = False
            truncated = self.current_steps >= self.max_steps
            info = {"reason": f"color_{current_color}_skipped_no_moves"}
            
            self.time_stats["block1_init_legal"] += (time.perf_counter() - t_start)
            self._write_time_stats()
            return obs, reward, terminated, truncated, info
        # legal_moves = self.state.generate_legal_moves(current_color) # 這一步也不要了應該是可以
        self.time_stats["block1_init_legal"] += (time.perf_counter() - t_start)

        # --- [區塊 2: 最大步數 Timeout 檢查] ---
        t_start = time.perf_counter()
        if self.current_steps >= self.max_steps:
            print(f"--- self.current_steps >= self.max_steps ---")
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)

            base = self._final_reward(left_total, left_each)
            reward = base - 100.0 

            current_color = self.colors[self.current_color_index]
            legal_moves = self.state.generate_legal_moves(current_color)
            self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(legal_moves)
            obs = self._get_obs(self.current_mask)

            terminated = False
            truncated = True
            info = {
                "reason": "max_steps_timeout",
                "left_total": left_total,
                "left_blue": left_each[BLUE],
                "left_yellow": left_each[YELLOW],
                "left_red": left_each[RED],
                "left_green": left_each[GREEN],
                "empty_cells": self._compute_empty_board_cells(),
            }
            self.time_stats["block2_timeout_check"] += (time.perf_counter() - t_start)
            self._write_time_stats()  # 覆寫輸出檔案
            return obs, reward, terminated, truncated, info
        self.time_stats["block2_timeout_check"] += (time.perf_counter() - t_start)

        
        # 應該是能夠保證此色有步可以走才會到這邊 (才會開始step)
        # # --- [區塊 4: 確保當前顏色有步可走] --- 
        # t_start = time.perf_counter()
        # self._ensure_current_color_has_moves()
        # self._last_legal_moves = legal_moves

        # if not legal_moves:
        #     left_total = self._compute_leftover_cells_total(self.state)
        #     left_each = self._compute_leftover_cells_each(self.state)
        #     final_reward = 0
        #     empty_cells = self._compute_empty_board_cells()
        #     current_color = self.colors[self.current_color_index]
        #     legal_moves = self.state.generate_legal_moves(current_color)
        #     self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(legal_moves)
        #     obs = self._get_obs(self.current_mask)
        #     terminated = False 
        #     truncated = False
        #     info = {
        #         "reason": "no_legal_moves_2",
        #         "left_total": left_total,
        #         "left_blue": left_each[BLUE],
        #         "left_yellow": left_each[YELLOW],
        #         "left_red": left_each[RED],
        #         "left_green": left_each[GREEN],
        #         "empty_cells": empty_cells,
        #     }
        #     self.time_stats["block4_ensure_color"] += (time.perf_counter() - t_start)
        #     self._write_time_stats()  # 覆寫輸出檔案
        #     return obs, final_reward, terminated, truncated, info
        # self.time_stats["block4_ensure_color"] += (time.perf_counter() - t_start)

        # --- [區塊 5: 取得 Move 與 偵錯檢查] ---
        t_start = time.perf_counter()
        move = self.current_padded_moves[action]
        if move is None:
            print(f"--- 偵錯資訊 ---")
            print(f"當前玩家顏色: {current_color}")
            print(f"模型選擇的 Action ID: {action}")
            print(f"該位置的遮罩狀態: {self.current_mask[action]}")
            self.state.print_board()
            raise RuntimeError(f"PPO選到了被遮罩的無效動作! Action ID: {action}")
        self.time_stats["block5_get_move"] += (time.perf_counter() - t_start)
        
        # --- [區塊 6: 落子前角落計算] ---
        t_start = time.perf_counter()
        all_colors = [1, 2, 3, 4]
        before_corners_count = 0
        for c in all_colors:
            before_corners_count += len(self.state.get_color_corners(c))
        self.time_stats["block6_corner_before"] += (time.perf_counter() - t_start)
        
        # --- [區塊 7: 執行落子 (Apply Move)] ---
        t_start = time.perf_counter()
        new_state = self.state.apply_move(
            current_color,
            move["shape"],
            move["x"],
            move["y"],
            move["piece"],
        )
        self.time_stats["block7_apply_move"] += (time.perf_counter() - t_start)

        # --- [區塊 8: 落子後角落與 Reward 計算] ---
        t_start = time.perf_counter()
        after_corners_count = 0
        for c in all_colors:
            after_corners_count += len(new_state.get_color_corners(c))

        space_diff = after_corners_count - before_corners_count
        self.state = new_state
        reward_size = len(move["shape"]) / 5 * 0.01 
        reward_space = space_diff * 0.02
        reward_crisis = -0.5 if after_corners_count < 8 else 0.0

        step_reward = reward_size + reward_space + reward_crisis
        step_reward = 0
        self.time_stats["block8_corner_after_calc"] += (time.perf_counter() - t_start)

        # --- [區塊 9: 換色、下一手 Mask 與常規返回 (終局函數高度細分版)] ---
        
        # 9-1-a. 檢查是否四色都沒步可走 (呼叫外部或內部核心邏輯)
        t_sub = time.perf_counter()
        is_no_moves = not self._any_legal_moves_for_any_color()
        self.time_stats["b9_1_a_any_legal_moves"] += (time.perf_counter() - t_sub)

        if is_no_moves:
            # --- 進入終局結算分支，細分內部所有函數 ---
            
            # 計算剩餘總方塊數
            t_sub = time.perf_counter()
            left_total = self._compute_leftover_cells_total(self.state)
            self.time_stats["b9_1_b_compute_left_total"] += (time.perf_counter() - t_sub)
            
            # 計算各色剩餘方塊數
            t_sub = time.perf_counter()
            left_each = self._compute_leftover_cells_each(self.state)
            self.time_stats["b9_1_c_compute_left_each"] += (time.perf_counter() - t_sub)
            
            # 計算基礎獎勵 (純數值運算，暫時歸在雜項)
            final_reward = self._final_reward(left_total, left_each)
            total_reward = step_reward + final_reward
            
            # 計算棋盤空格
            t_sub = time.perf_counter()
            empty_cells = self._compute_empty_board_cells()
            self.time_stats["b9_1_d_compute_empty_cell"] += (time.perf_counter() - t_sub)
            
            # 剩餘常規動作
            t_sub = time.perf_counter()
            self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(legal_moves)
            obs = self._get_obs(self.current_mask)
            terminated = True
            truncated = False
            info = {
                "reason": "no_legal_moves_all_3",
                "left_total": left_total,
                "left_blue": left_each[BLUE],
                "left_yellow": left_each[YELLOW],
                "left_red": left_each[RED],
                "left_green": left_each[GREEN],
                "empty_cells": empty_cells,
            }
            self.time_stats["b9_5_get_obs_and_misc"] += (time.perf_counter() - t_sub)
            self._write_time_stats()
            return obs, total_reward, terminated, truncated, info

        # 9-2. 遊戲繼續：換到下一個顏色
        t_sub = time.perf_counter()
        self._switch_color()
        self.time_stats["b9_2_switch_color"] += (time.perf_counter() - t_sub)

        # 9-3. 為「下一手」準備，生成下一個顏色的合法步
        t_sub = time.perf_counter()
        next_color = self.colors[self.current_color_index]
        next_legal_moves = self.state.generate_legal_moves(next_color)
        self.time_stats["b9_3_gen_next_legal_moves"] += (time.perf_counter() - t_sub)
        
        # 9-4. 計算下一手的 mask 與 padded_moves
        t_sub = time.perf_counter()
        self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(next_legal_moves)
        self.time_stats["b9_4_get_padded_and_mask"] += (time.perf_counter() - t_sub)
        
        # 9-5. 構造回傳的 obs 與其餘常規返回設定
        t_sub = time.perf_counter()
        obs = self._get_obs(self.current_mask)
        reward = step_reward
        terminated = False
        truncated = self.current_steps >= self.max_steps
        info = {}
        self.time_stats["b9_5_get_obs_and_misc"] += (time.perf_counter() - t_sub)
        
        # 覆寫輸出檔案並返回
        self._write_time_stats()
        return obs, reward, terminated, truncated, info

    # -----------------------
    # Observation / Reward
    # -----------------------

    def _get_obs(self, action_mask_list):
        """
        obs = {
            "board": (H,W) 0/1/2/3/4/5,
            "remaining_blue":   (num_pieces,),
            "remaining_yellow": (num_pieces,),
            "remaining_red":    (num_pieces,),
            "remaining_green":  (num_pieces,),
            "turn": (1,) 0~3
        }
        """
        assert self.state is not None
        # 將 True/False 轉為 1/0 的 numpy array
        mask_array = np.array(action_mask_list, dtype=np.int8)

        board_arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                v = self.state.board[y][x]
                if v == BLUE:
                    board_arr[y, x] = 1
                elif v == YELLOW:
                    board_arr[y, x] = 2
                elif v == RED:
                    board_arr[y, x] = 3
                elif v == GREEN:
                    board_arr[y, x] = 4
                else:
                    board_arr[y, x] = 0

        remaining_blue = np.zeros((self.num_pieces,), dtype=np.int8)
        remaining_yellow = np.zeros((self.num_pieces,), dtype=np.int8)
        remaining_red = np.zeros((self.num_pieces,), dtype=np.int8)
        remaining_green = np.zeros((self.num_pieces,), dtype=np.int8)

        rem_b = self.state.remaining_pieces[BLUE]
        rem_y = self.state.remaining_pieces[YELLOW]
        rem_r = self.state.remaining_pieces[RED]
        rem_g = self.state.remaining_pieces[GREEN]

        for i, name in enumerate(self.piece_names):
            if name in rem_b:
                remaining_blue[i] = 1
            if name in rem_y:
                remaining_yellow[i] = 1
            if name in rem_r:
                remaining_red[i] = 1
            if name in rem_g:
                remaining_green[i] = 1

        turn_flag = np.array(
            [self.current_color_index], dtype=np.int8
        )  # 0=BLUE, 1=YELLOW, 2=RED, 3=GREEN

        return {
            "board": board_arr,
            "remaining_blue": remaining_blue,
            "remaining_yellow": remaining_yellow,
            "remaining_red": remaining_red,
            "remaining_green": remaining_green,
            "turn": turn_flag,
            "action_mask": mask_array,
        }

    def _compute_leftover_cells_total(self, state: GameState) -> int:
        """
        計算四個顏色剩餘格數總和
        """
        total = 0
        for color in self.colors:
            for name in state.remaining_pieces[color]:
                total += len(BASE_PIECES[name])
        return total

    def _compute_leftover_cells_each(self, state: GameState):
        """
        回傳每個顏色剩餘格數：{color: left_cells}
        """
        res = {}
        for color in self.colors:
            cnt = 0
            for name in state.remaining_pieces[color]:
                cnt += len(BASE_PIECES[name])
            res[color] = cnt
        return res
    def _final_reward(self, left_total: int, left_each: dict[int, int]) -> float:
        """
        left_total: 所有玩家剩餘的棋子方格總數 (以4人制標準 Blokus 為例，總方格數為 89 * 4 = 356)
        left_each: 每個 color 剩餘的方格數，例如 {0: 10, 1: 12, 2: 8, 3: 15}
        """

        # 標準指數衰減
        # k 值決定了曲線的彎曲程度。k=0.04 可以讓 100 左右完美收尾在 -2
        k = 0.04 
        base_reward = -2.0 * (1.0 - math.exp(-k * left_total))

        # 棋子剩得越少，分數越高。完美清空為 +1.0
        base_reward = 1.0 - (left_total / 80)
    

        # print(f"base_reward = {base_reward}")
        # 如果你想給「完全清空 (0)」一個額外的完美加成 (Bonus)
        if left_total == 0:
            base_reward += 0.5  # 總分變成 0.5

        # ==========================================
        # 2) 團隊合作平衡懲罰 (Cooperative Penalty)
        # ==========================================
        # 目的：避免單一角色肥大、其他隊友被卡死的自私行為
        scores = list(left_each.values())
        
        if len(scores) > 1:
            # 計算各玩家剩餘棋子數的標準差
            std_dev = np.std(scores)
            
            # 或者是計算最大與最小的差距 (Max-Min Difference)
            # diff = max(scores) - min(scores)
            
            # 將懲罰項縮放到一個合理的範圍 (例如最大扣 0.3)
            # 標準差越大（越不平均），扣分越多
            # 假設極端不平均時標準差可能到 40~50，我們將其除以一個基數
            balance_penalty = (std_dev / 20.0) * 0.3
            balance_penalty = min(balance_penalty, 0.4) # 設個天花板
        else:
            balance_penalty = 0.0
            
        # ==========================================
        # 3) 最終結算
        # ==========================================
        final_score = base_reward - balance_penalty
        
        return float(final_score)
    def old_final_reward(self, left_total: int, left_each: dict[int, int]) -> float:
        # 1) 先算 base
        if left_total == 0:
            base = 200.0
        elif left_total <= 20:
            base = 100.0
        elif left_total <= 40:
            base = 60.0
        elif left_total <= 60:
            base = 20.0
        elif left_total <= 80:
            base = -0.3 * left_total
        elif left_total <= 110:
            base = -0.8 * left_total
        else:
            base = -1.3 * left_total
        return base

    def _select_candidate_moves(self, legal_moves: list[dict]) -> list[dict]:
        if len(legal_moves) <= self.max_candidates:
            return legal_moves

        def move_score(m: dict) -> float:
            shape = m["shape"]  # list[(dx, dy)]
            size = len(shape)
            return size

        scored = sorted(legal_moves, key=move_score, reverse=True)
        return scored[: self.max_candidates]

    def _map_action_to_move(self, action: int, candidate_moves: list[dict]) -> dict | None:
        if action < 0 or action >= len(candidate_moves):
            return None
        return candidate_moves[action]
    
    def _get_padded_moves_and_mask(self, legal_moves: list[dict]) -> tuple[list[dict], list[bool]]:  
        """
        超高速優化版（固定動作空間）：
        移除沉重的排序邏輯，改用 O(N) 的雜湊映射直接將合法動作對齊到固定的 Action ID 上。
        """
        # 1. 建立一個長度與全域動作空間完全相同、預設皆為 False 的遮罩
        action_mask = [False] * self.total_action_space_size
        
        # 2. 將當前盤面算出來的合法動作，透過字典一瞬間找出對應的固定 Action ID 並解鎖 (設為 True)
        for m in legal_moves:
            key = (m["piece"], m["o_idx"], m["x"], m["y"])
            
            # 安全防護：確保該動作存在於全域空間中（理論上一定在）
            if key in self.MOVE_TO_ACTION_ID:
                action_id = self.MOVE_TO_ACTION_ID[key]
                action_mask[action_id] = True
                
        # 3. 完美的常規返回：
        # 這裡不需要返回帶有 None 的列表了，因為 self.current_padded_moves 
        # 永遠指向量不變的 self.GLOBAL_CANDIDATE_MOVES。
        return self.GLOBAL_CANDIDATE_MOVES, action_mask
    
    def _get_padded_moves_and_mask_old(self, legal_moves: list[dict]) -> tuple[list[dict | None], list[bool]]:  
        """
        將合法動作填入固定大小的 slots 中，並產生對應的 Action Mask。
        確保同一個特徵的動作盡可能落在固定的語義位置。
        """
        # 1. 為了保持某種程度的語義一致性，我們使用全域固定的 key 排序（而非動態的盤面分數）
        # 例如：依據方塊名稱字串、x座標、y座標排序。這樣相同的動作在不同回合會排在相對一致的位置。
        fixed_sorted_moves = sorted(
            legal_moves, 
            key=lambda m: (m["piece"], m["x"], m["y"])
        )
        
        candidate_moves = []
        action_mask = []
        
        for i in range(self.max_candidates):
            if i < len(fixed_sorted_moves):
                candidate_moves.append(fixed_sorted_moves[i])
                action_mask.append(True)  # 合法動作
            else:
                candidate_moves.append(None)
                action_mask.append(False) # 遮罩掉的無效動作
                
        return candidate_moves, action_mask
    # -----------------------
    # 顏色輪轉 & 合法步檢查
    # -----------------------

    def _any_legal_moves_for_any_color(self) -> bool:
        """
        優化版：利用快取狀態判斷是否還有任何顏色有合法步。
        時間複雜度從 O(顏色數 * 窮舉所有步) 降到 O(顏色數)，幾乎是瞬間完成！
        """
        assert self.state is not None
        # 直接檢查快取字典，只要還有任何一個顏色是 True，遊戲就還沒結束
        return any(self.state.color_has_moves[c] for c in self.state.colors)
    
    def _ensure_current_color_has_moves(self):
        """
        超高速優化版：利用 self.color_has_moves 快取字典，
        快速跳過已知沒步可走的玩家，避免重複進行沉重的合法步窮舉。
        """
        assert self.state is not None

        for _ in range(len(self.colors)):
            current_color = self.colors[self.current_color_index]
            
            # 【核心優化點 1】: 如果快取直接記錄這顏色早就沒步了，連算都不用算，直接跳下一色！
            if not self.state.color_has_moves[current_color]:
                self.current_color_index = (self.current_color_index + 1) % len(self.colors)
                continue
                
            # 【核心優化點 2】: 只有在快取認為「它還有步」時，我們才呼叫 generate_legal_moves。
            # 因為你在優化後的 generate_legal_moves 裡已經會自動更新 self.color_has_moves，
            # 所以這裡只要拿到 moves，就能100%確認生死。
            legal_moves = self.state.generate_legal_moves(current_color)
            
            if legal_moves:
                # 確定有步，目前的 current_color_index 是合法的，收工！
                return
                
            # 如果算完發現其實沒步了（此時 generate_legal_moves 內部已將其設為 False）
            # 換下一色試試
            self.current_color_index = (self.current_color_index + 1) % len(self.colors)

    def _switch_color(self):
        """
        下一步換下一個顏色（實際 step 開頭會再檢查有沒有步）
        """
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)

    # -----------------------
    # Render（文字版）
    # -----------------------
    def _color_cell(self, ch: str) -> str:
        # ANSI background colors
        BG = {
        "B": "\033[44m", # 藍底
        "Y": "\033[43m", # 黃底
        "R": "\033[41m", # 紅底
        "G": "\033[42m", # 綠底
        "#": "\033[100m", # 灰底（障礙）
        ".": "\033[40m", # 黑底（空格）
        }
        RESET = "\033[0m"
        return f"{BG.get(ch, '')}{ch}{RESET}"
    def render(self):
        """
        'B' = BLUE, 'Y' = YELLOW, 'R' = RED, 'G' = GREEN, '#' = 障礙, '.' = 空
        """
        if self.state is None:
            print("<Env 未初始化，請先 reset()>")
            return

        print("Board:")
        for y in range(BOARD_SIZE):
            row = ""
            for x in range(BOARD_SIZE):
                v = self.state.board[y][x]
                if v == BLUE:
                    ch = "B"
                elif v == YELLOW:
                    ch = "Y"
                elif v == RED:
                    ch = "R"
                elif v == GREEN:
                    ch = "G"
                else:
                    ch = "."
                row += self._color_cell(ch)
            print(row)
        left_each = self._compute_leftover_cells_each(self.state)
        print(
            "Remaining:",
            f"B={left_each[BLUE]}",
            f"Y={left_each[YELLOW]}",
            f"R={left_each[RED]}",
            f"G={left_each[GREEN]}",
        )
        print("Current turn:", ["BLUE", "YELLOW", "RED", "GREEN"][self.current_color_index])
        print()
