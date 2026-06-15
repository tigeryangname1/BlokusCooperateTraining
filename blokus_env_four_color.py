import numpy as np
import gymnasium as gym
import math

from gymnasium import spaces
from blokus import GameState, BOARD_SIZE, BLUE, YELLOW, RED, GREEN, BASE_PIECES


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
        max_candidates: int = 512,
    ):
        super().__init__()
        
        # 四個顏色都由同一個 agent 控制
        self.colors = [BLUE, YELLOW, RED, GREEN]
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

        # 【注意】請刪除原本末尾那段錯誤的 self.observation_space 重新賦值程式碼

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
    
    def step(self, action: int):
        assert self.state is not None, "請先呼叫 reset()"
        self.current_steps += 1
        self.state
        current_color = self.colors[self.current_color_index]
        legal_moves = self.state.generate_legal_moves(current_color)

        if self.current_steps >= self.max_steps:
            print(f"--- self.current_steps >= self.max_steps ---")
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)

            # 用和正常終局一樣的計算，再多扣一點
            base = self._final_reward(left_total, left_each)
            reward = base - 100.0 # 額外再罰 50，就表達「拖到timeout更糟」

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
            return obs, reward, terminated, truncated, info
        # 如果四色都沒步可走 -> 結束
        if not self._any_legal_moves_for_any_color():
            # print(f"--- not self._any_legal_moves_for_any_color() ---")
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = self._final_reward(left_total, left_each)
            empty_cells = self._compute_empty_board_cells()
            self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(legal_moves)
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
            return obs, final_reward, terminated, truncated, info

        # 確保 current_color 有步可以走，如果沒有就跳到下一個有步的顏色
        self._ensure_current_color_has_moves()
        
        self._last_legal_moves = legal_moves

        # 若該色沒不可以走就跳到下一個step
        if not legal_moves:
            # print(f"--- not legal_moves ---")
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = 0
            empty_cells = self._compute_empty_board_cells()
            current_color = self.colors[self.current_color_index]
            legal_moves = self.state.generate_legal_moves(current_color)
            self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(legal_moves)
            obs = self._get_obs(self.current_mask)
            terminated = False #繼續跑
            truncated = False
            info = {
                "reason": "no_legal_moves_2",
                "left_total": left_total,
                "left_blue": left_each[BLUE],
                "left_yellow": left_each[YELLOW],
                "left_red": left_each[RED],
                "left_green": left_each[GREEN],
                "empty_cells": empty_cells,
            }
            return obs, final_reward, terminated, truncated, info
        # print(f"action = {action}")
        move = self.current_padded_moves[action]
        # 理論上因為有 Action Mask，這裡的 move 絕對不會是 None。
        # 如果是 None，代表環境或 Mask 套用邏輯有 Bug。
        if move is None:
            # 這裡建議加上列印資訊，能幫你瞬間看清是哪裡不對
            print(f"--- 偵錯資訊 ---")
            print(f"當前玩家顏色: {current_color}")
            print(f"當前合法步數量: {len(legal_moves)}")
            print(f"模型選擇的 Action ID: {action}")
            print(f"該位置的遮罩狀態: {self.current_mask[action]}")
            self.state.print_board()
            raise RuntimeError(f"PPO選到了被遮罩的無效動作! Action ID: {action}")
        
         # 1. 【落子前】計算全團隊 4 個顏色的總可用角落數
        # 假設顏色代號是 1, 2, 3, 4 (或是 0, 1, 2, 3，依你的設計調整)
        all_colors = [1, 2, 3, 4]
        before_corners_count = 0
        for c in all_colors:
            # 呼叫你本體現有的 get_color_corners，並計算集合的長度
            before_corners_count += len(self.state.get_color_corners(c))
        
        # 套用這一步
        new_state = self.state.apply_move(
            current_color,
            move["shape"],
            move["x"],
            move["y"],
            move["piece"],
        )
        # 在 apply_move 後
        
        # 3. 【落子後】計算全團隊 4 個顏色的總可用角落數
        after_corners_count = 0
        for c in all_colors:
            after_corners_count += len(new_state.get_color_corners(c))

        # 4. 【計算 Step Reward】
        # 空間增減量 = 落子後總數 - 落子前總數
        space_diff = after_corners_count - before_corners_count

        self.state = new_state
        # 方塊大小分數
        reward_size = len(move["shape"]) / 5 * 0.01 # 四色共享 reward
        # 角落增減分數
        reward_space = space_diff * 0.02

        # (C) 懲罰項：如果全隊角落總數降得太低（代表有人被徹底堵死）
        # 假設開局大家角很多，如果總角數低於某個閾值，給予集體警告
        reward_crisis = -0.5 if after_corners_count < 8 else 0.0

        step_reward = reward_size + reward_space + reward_crisis
        step_reward = 0
        # 5) 檢查是否四色都沒步可走
        if not self._any_legal_moves_for_any_color():
            # print(f"--- not self._any_legal_moves_for_any_color() ---")
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = self._final_reward(left_total, left_each)
            total_reward = step_reward + final_reward
            empty_cells = self._compute_empty_board_cells()
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
            return obs, total_reward, terminated, truncated, info

        # 6) 遊戲繼續：換到下一個顏色
        self._switch_color()

        # 7) 為「下一手」準備 Observation 與 Action Mask
        next_color = self.colors[self.current_color_index]
        next_legal_moves = self.state.generate_legal_moves(next_color)
        # 計算下一手的 mask
        self.current_padded_moves, self.current_mask = self._get_padded_moves_and_mask(next_legal_moves)
        
        # 構造回傳的 obs (必須包含下一手的 action_mask)
        obs = self._get_obs(self.current_mask)
        
        reward = step_reward
        terminated = False
        truncated = self.current_steps >= self.max_steps
        info = {}
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
    
    def _get_padded_moves_and_mask(self, legal_moves: list[dict]) -> tuple[list[dict | None], list[bool]]:  
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
        是否還有任何顏色有合法步
        """
        assert self.state is not None
        for c in self.colors:
            if self.state.generate_legal_moves(c):
                return True
        return False

    def _ensure_current_color_has_moves(self):
        """
        如果 current_color 沒步、但其他顏色有，就跳到下一個有步的顏色。
        """
        assert self.state is not None

        for _ in range(len(self.colors)):
            current_color = self.colors[self.current_color_index]
            legal_moves = self.state.generate_legal_moves(current_color)
            if legal_moves:
                return
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
