import numpy as np
import gymnasium as gym
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
        min_obstacles: int = 0,
        max_obstacles: int = 0,
    ):
        super().__init__()

        # 四個顏色都由同一個 agent 控制
        self.colors = [BLUE, YELLOW, RED, GREEN]
        self.current_color_index = 0  # 0=BLUE, 1=YELLOW, 2=RED, 3=GREEN
        self.max_candidates = max_candidates

        # 隨機障礙設定
        self.min_obstacles = 0
        self.max_obstacles = 0
        self.obstacle_value = 9
        self.blocked_cells: set[tuple[int, int]] = set()

        # 棋子順序（四色共用同一組）
        self.piece_names = sorted(BASE_PIECES.keys())
        self.num_pieces = len(self.piece_names)

        # === Observation Space ===
        # board: (H,W) 0/1/2/3/4/5
        #   0 = 空
        #   1 = BLUE
        #   2 = YELLOW
        #   3 = RED
        #   4 = GREEN
        #   5 = 障礙
        # remaining_*: (num_pieces,) 0/1
        # turn: (1,) 0~3
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=5,
                    shape=(BOARD_SIZE, BOARD_SIZE),
                    dtype=np.int8,
                ),
                "remaining_blue": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_pieces,),
                    dtype=np.int8,
                ),
                "remaining_yellow": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_pieces,),
                    dtype=np.int8,
                ),
                "remaining_red": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_pieces,),
                    dtype=np.int8,
                ),
                "remaining_green": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_pieces,),
                    dtype=np.int8,
                ),
                "turn": spaces.Box(
                    low=0,
                    high=3,
                    shape=(1,),
                    dtype=np.int8,
                ),
            }
        )

        # === Action Space ===
        # 和前面一樣：選擇候選合法步的 index
        self.action_space = spaces.Discrete(self.max_candidates)

        # 內部狀態
        self.state: GameState | None = None
        self._last_legal_moves: list[dict] = []
        self.current_steps = 0
        self.max_steps = 600  # 安全上限，避免極端長局

    # -----------------------
    # Gym API
    # -----------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = GameState()
        self._last_legal_moves = []
        self.blocked_cells.clear()
        self.current_steps = 0
        self.current_color_index = 0  # 從 BLUE 開始

        self._sample_blocked_cells()

        obs = self._get_obs()
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
    def step(self, action: int):
        assert self.state is not None, "請先呼叫 reset()"
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)

            # 用和正常終局一樣的計算，再多扣一點
            base = self._final_reward(left_total, left_each)
            reward = base - 100.0 # 額外再罰 50，就表達「拖到timeout更糟」

            obs = self._get_obs()
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
        # 1) 如果四色都沒步可走 -> 結束
        if not self._any_legal_moves_for_any_color():
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = self._final_reward(left_total, left_each)
            empty_cells = self._compute_empty_board_cells()
            obs = self._get_obs()
            terminated = True
            truncated = False
            info = {
                "reason": "no_legal_moves_all",
                "left_total": left_total,
                "left_blue": left_each[BLUE],
                "left_yellow": left_each[YELLOW],
                "left_red": left_each[RED],
                "left_green": left_each[GREEN],
                "empty_cells": empty_cells,
            }
            return obs, final_reward, terminated, truncated, info

        # 2) 確保 current_color 有步可以走，如果沒有就跳到下一個有步的顏色
        self._ensure_current_color_has_moves()

        current_color = self.colors[self.current_color_index]
        legal_moves = self.state.generate_legal_moves(current_color)
        self._last_legal_moves = legal_moves

        # 再次防呆：如果還是沒有（理論上代表四色都沒步了）
        if not legal_moves:
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = self._final_reward(left_total, left_each)
            empty_cells = self._compute_empty_board_cells()
            obs = self._get_obs()
            terminated = True
            truncated = False
            info = {
                "reason": "no_legal_moves_all",
                "left_total": left_total,
                "left_blue": left_each[BLUE],
                "left_yellow": left_each[YELLOW],
                "left_red": left_each[RED],
                "left_green": left_each[GREEN],
                "empty_cells": empty_cells,
            }
            return obs, final_reward, terminated, truncated, info

        # 3) 把 legal_moves 壓到 max_candidates
        candidate_moves = self._select_candidate_moves(legal_moves)
        move = self._map_action_to_move(action, candidate_moves)

        if move is None:
            # 選到無效動作：小懲罰，但不結束，也先不換色
            obs = self._get_obs()
            reward = -1.0
            terminated = False
            truncated = self.current_steps >= self.max_steps
            info = {"invalid_action": True}
            return obs, reward, terminated, truncated, info
        # 在 apply_move 前
        legal_before = len(self.state.generate_legal_moves(current_color))

        # 4) 套用這一步
        new_state = self.state.apply_move(
            current_color,
            move["shape"],
            move["x"],
            move["y"],
            move["piece"],
        )
        self.state = new_state

        step_reward = len(move["shape"]) * 1 # 四色共享 reward
        # 在 apply_move 後
        legal_after = len(new_state.generate_legal_moves(current_color))
        # shaping reward（關鍵）
        step_reward += 0.1 * (legal_after - legal_before)
        # 5) 檢查是否四色都沒步可走
        if not self._any_legal_moves_for_any_color():
            left_total = self._compute_leftover_cells_total(self.state)
            left_each = self._compute_leftover_cells_each(self.state)
            final_reward = self._final_reward(left_total, left_each)
            total_reward = step_reward + final_reward
            empty_cells = self._compute_empty_board_cells()

            obs = self._get_obs()
            terminated = True
            truncated = False
            info = {
                "reason": "no_legal_moves_all",
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

        obs = self._get_obs()
        reward = step_reward
        terminated = False
        truncated = self.current_steps >= self.max_steps
        info = {}
        return obs, reward, terminated, truncated, info

    # -----------------------
    # Observation / Reward
    # -----------------------

    def _get_obs(self):
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

        board_arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if (x, y) in self.blocked_cells:
                    board_arr[y, x] = 5  # 障礙
                else:
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
        # 1) 先算 base
        if left_total == 0:
            base = 9999.0
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
    # 隨機障礙
    # -----------------------

    def _sample_blocked_cells(self):
        """
        隨機生成障礙格，避免蓋到四色起始角
        """
        assert self.state is not None

        forbidden = set()
        for c in self.colors:
            forbidden.update(self.state.start_corners[c])

        all_cells = [
            (x, y)
            for y in range(BOARD_SIZE)
            for x in range(BOARD_SIZE)
            if (x, y) not in forbidden
        ]

        if not all_cells:
            return

        n_min = max(0, self.min_obstacles)
        n_max = max(n_min, self.max_obstacles)
        n_obstacles = int(self.np_random.integers(n_min, n_max + 1))

        n_obstacles = min(n_obstacles, len(all_cells))
        if n_obstacles <= 0:
            return

        idxs = self.np_random.choice(len(all_cells), size=n_obstacles, replace=False)
        self.blocked_cells = {all_cells[i] for i in idxs}

        for (x, y) in self.blocked_cells:
            self.state.board[y][x] = self.obstacle_value

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
                if (x, y) in self.blocked_cells:
                    ch = "#"
                else:
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
