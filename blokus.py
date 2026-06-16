import os
from colorama import init
from copy import deepcopy

init()
BOARD_SIZE = 20

EMPTY = 0
BLUE  = 1   # 我方顏色1
YELLOW= 2   # 我方顏色2
RED   = 3   # 敵方顏色1
GREEN = 4   # 敵方顏色2

ALLY_COLORS = {BLUE, YELLOW}
RESET = "\033[0m"

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
DIM = "\033[2m"

BG_MAGENTA = "\033[45m"   # 用來顯示 AI 建議 overlay
FG_GRAY = "\033[90m"

# 前景色
FG_BLACK = "\033[30m"
FG_WHITE = "\033[97m"

# 背景色
BG_BLUE   = "\033[44m"
BG_RED    = "\033[41m"
BG_YELLOW = "\033[43m"
BG_GREEN  = "\033[42m"
COLOR_NAME = {
    EMPTY: " . ",
    BLUE:   f"{BG_BLUE}{FG_WHITE} B {RESET}",
    YELLOW: f"{BG_YELLOW}{FG_BLACK} Y {RESET}",
    RED:    f"{BG_RED}{FG_WHITE} R {RESET}",
    GREEN:  f"{BG_GREEN}{FG_BLACK} G {RESET}",
}
def clear_screen():
    # Windows 用 cls，Linux / Mac 用 clear
    os.system('cls' if os.name == 'nt' else 'clear')
    
# ==============================
# Polyomino 定義區（Blokus 21 塊）
# ==============================

def normalize(shape):
    """把座標平移，讓最小 x, y 變成 (0,0)，方便去重"""
    min_x = min(x for x, y in shape)
    min_y = min(y for x, y in shape)
    return frozenset((x - min_x, y - min_y) for x, y in shape)

def rotate(shape):
    """順時針旋轉 90 度: (x, y) -> (y, -x)"""
    return normalize({(y, -x) for (x, y) in shape})

def flip(shape):
    """左右翻轉: (x, y) -> (-x, y)"""
    return normalize({(-x, y) for (x, y) in shape})

def all_orientations(base_shape):
    """給一個基本形狀，產生所有獨特的旋轉＋翻轉形態"""
    shapes = set()
    s = normalize(base_shape)
    for _ in range(4):
        shapes.add(s)
        shapes.add(flip(s))
        s = rotate(s)
    return list(shapes)

# Blokus 使用的 21 個 polyomino
# 1: monomino
# 2: domino
# 3: triomino（2 種）
# 4: tetromino（5 種：I, O, T, S, L）
# 5: pentomino（12 種：F, I, L, N, P, T, U, V, W, X, Y, Z）
# 座標只要拓樸形狀正確即可，實際方向會透過 all_orientations 生成。
BASE_PIECES = {
    # --- 1-omino ---
    "P1": {(0, 0)},  # monomino

    # --- 2-omino ---
    "P2": {(0, 0), (1, 0)},  # domino

    # --- 3-omino ---
    # I 形
    "P3_I": {(0, 0), (1, 0), (2, 0)},
    # L 形
    "P3_L": {(0, 0), (0, 1), (1, 0)},

    # --- 4-omino ---
    # I 形
    "P4_I": {(0, 0), (1, 0), (2, 0), (3, 0)},
    # O 形（正方形）
    "P4_O": {(0, 0), (1, 0), (0, 1), (1, 1)},
    # T 形
    "P4_T": {(0, 0), (1, 0), (2, 0), (1, 1)},
    # S 形（Z/S 型四格）
    "P4_S": {(0, 1), (1, 1), (1, 0), (2, 0)},
    # L 形（4 格）
    "P4_L": {(0, 0), (0, 1), (0, 2), (1, 0)},

    # --- 5-omino (12 種 pentomino) ---
    # I pentomino
    "P5_I": {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)},
    # L pentomino（長 4 + 一個側邊）
    "P5_L": {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)},
    # P pentomino（2x2 方塊 + 一格）
    "P5_P": {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)},
    # N pentomino（兩排交錯）
    "P5_N": {(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)},
    # T pentomino（上三格＋下面中間两格）
    "P5_T": {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)},
    # U pentomino（中空）
    "P5_U": {(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
    # V pentomino（L 形延伸）
    "P5_V": {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)},
    # W pentomino（階梯三段）
    "P5_W": {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
    # X pentomino（十字）
    "P5_X": {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
    # Y pentomino（一條 4 格 + 分支）
    "P5_Y": {(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)},
    # Z pentomino（彎曲 Z）
    "P5_Z": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)},
    # F pentomino（不對稱）
    "P5_F": {(1, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
}

# 預先把每種基底形狀展開成所有 orientation
ALL_PIECES = {
    name: all_orientations(shape)
    for name, shape in BASE_PIECES.items()
}

print("=== 每個棋子的轉向數量清單 ===")
for name, orientations in ALL_PIECES.items():
    print(f"棋子名稱: {name:<5} | 轉向數量: {len(orientations)}")
input()
# ==============================
# 遊戲狀態
# ==============================

class GameState:
    def __init__(self):
        self.board = [[EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.remaining_pieces = {
            BLUE:   set(ALL_PIECES.keys()),
            YELLOW: set(ALL_PIECES.keys()),
            RED:    set(ALL_PIECES.keys()),
            GREEN:  set(ALL_PIECES.keys()),
        }
        # 初始角落（Blokus 標準 4 角）
        self.start_corners = {
            BLUE:   [(0, 0)],
            YELLOW: [(BOARD_SIZE-1, BOARD_SIZE-1)],
            RED:    [(0, BOARD_SIZE-1)],
            GREEN:  [(BOARD_SIZE-1, 0)],
        }
        self.colors = [BLUE, YELLOW, RED, GREEN]
        self.color_has_moves = {color: True for color in self.colors}
    # ---- 基礎工具 ----

    def in_bounds(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def print_board(self, last_cells=None, suggest_cells=None):
        """
        last_cells: set[(x,y)]  剛放上的棋子格子 -> 用粗體/底線高亮
        suggest_cells: set[(x,y)] AI 建議要下的位置 -> 空格用紫底 ? 顯示
        """
        last_cells = last_cells or set()
        suggest_cells = suggest_cells or set()

        def render_cell(x, y):
            cell = self.board[y][x]

            # 1) AI 建議 overlay：只在「空格」上顯示
            if cell == EMPTY and (x, y) in suggest_cells:
                return f"{BG_MAGENTA}{FG_WHITE} ? {RESET}"

            # 2) 剛下的棋子高亮：同色，但加粗/底線
            if cell != EMPTY and (x, y) in last_cells:
                # 用原本 COLOR_NAME 的底色，但再套 BOLD+UNDERLINE
                # COLOR_NAME[cell] 形如: "\033[44m\033[97m B \033[0m"
                # 我們在前面加 BOLD+UNDERLINE，並在末尾 RESET（原本已有 RESET）
                base = COLOR_NAME[cell]
                return f"{BOLD}{UNDERLINE}{base}{RESET}"

            # 3) 一般格子
            return COLOR_NAME[cell]

        # X 軸標頭
        print("    ", end="")
        for x in range(BOARD_SIZE):
            print(f"{x:2d} ", end="")
        print()

        for y in range(BOARD_SIZE):
            print(f"{y:2d}  ", end="")
            for x in range(BOARD_SIZE):
                print(render_cell(x, y), end="")
            print()

    def any_piece_on_board(self, color):
        return any(self.board[y][x] == color
                   for y in range(BOARD_SIZE)
                   for x in range(BOARD_SIZE))

    # ---- 角落計算 ----

    def get_color_corners(self, color):
        """該色目前可落子的角位置集合"""
        # 若尚未落子，回傳起始角
        if not self.any_piece_on_board(color):
            return set(self.start_corners[color])

        corners = set()
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] != color:
                    continue
                # 試四個對角
                for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                    nx, ny = x + dx, y + dy
                    if not self.in_bounds(nx, ny):
                        continue
                    if self.board[ny][nx] != EMPTY:
                        continue
                    # 邊不能相接
                    side_neighbors = [
                        (nx+1, ny),
                        (nx-1, ny),
                        (nx, ny+1),
                        (nx, ny-1),
                    ]
                    if any(
                        self.in_bounds(sx, sy) and self.board[sy][sx] == color
                        for sx, sy in side_neighbors
                    ):
                        continue
                    corners.add((nx, ny))
        return corners

    # ---- 合法性檢查 ----

    def is_legal_move(self, color, shape_cells, origin_x, origin_y):
        """
        shape_cells: frozenset of (dx,dy)
        origin + shape_cells = 要蓋的絕對座標
        """
        has_corner_touch = False

        # 先檢查不出界 & 不重疊
        for dx, dy in shape_cells:
            x, y = origin_x + dx, origin_y + dy
            if not self.in_bounds(x, y):
                return False
            if self.board[y][x] != EMPTY:
                return False

        first_move = not self.any_piece_on_board(color)

        for dx, dy in shape_cells:
            x, y = origin_x + dx, origin_y + dy

            # 邊接觸不能有同色
            for sx, sy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x+sx, y+sy
                if self.in_bounds(nx, ny) and self.board[ny][nx] == color:
                    return False

            # 檢查角接觸（非第一手時需要至少一個）
            for cx, cy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = x+cx, y+cy
                if self.in_bounds(nx, ny) and self.board[ny][nx] == color:
                    has_corner_touch = True

        if first_move:
            # 第一手：必須覆蓋起始角
            starts = set(self.start_corners[color])
            covered = {(origin_x + dx, origin_y + dy) for dx, dy in shape_cells}
            return any(p in covered for p in starts)

        # 之後的手：必須有角接觸
        return has_corner_touch

    # ---- apply move ----

    def apply_move(self, color, shape_cells, origin_x, origin_y, piece_name):
        new_state = deepcopy(self)
        # new_state = (self)
        for dx, dy in shape_cells:
            x, y = origin_x + dx, origin_y + dy
            new_state.board[y][x] = color
        if piece_name in new_state.remaining_pieces[color]:
            new_state.remaining_pieces[color].remove(piece_name)
        return new_state

    # ---- 合法步產生 ----

    def generate_legal_moves(self, color):
        """
        優化版：在每次生成合法步時，動態更新該顏色的生存狀態。
        """
        # 額外優化：如果這個顏色之前就已經被判定沒步了，連算都不用算，直接回傳空列表
        if not self.color_has_moves[color]:
            return []

        moves = []
        corners = self.get_color_corners(color) # 注意：如果是讀取 state 請確認路徑
        
        if not corners:
            # 沒有角落代表死路一條，更新狀態為 False
            self.color_has_moves[color] = False
            return moves

        for piece_name in sorted(self.remaining_pieces[color]):
            orientations = ALL_PIECES[piece_name]
            # 【關鍵點】直接在這裡用 enumerate() 就能抓出固定的 o_idx！
            for o_idx, shape in enumerate(orientations):
                # 嘗試將 shape 的某一格對準某個 corner
                for (cx, cy) in corners:
                    for (dx, dy) in shape:
                        origin_x = cx - dx
                        origin_y = cy - dy
                        if self.is_legal_move(color, shape, origin_x, origin_y):
                            moves.append({
                                "piece": piece_name,
                                "shape": shape,
                                "x": origin_x,
                                "y": origin_y,
                                "o_idx": o_idx  # 增加第幾個轉向給PPO用
                            })
        
        # 【關鍵優化點】: 根據這次窮舉的結果，更新這個顏色的生死狀態
        # 如果 moves 是空的，代表此顏色正式宣告陣亡，以後不用再幫它算步了
        self.color_has_moves[color] = len(moves) > 0
        
        return moves
    # ---- 兩色角落計數 ----

    def count_ally_corners(self):
        total = 0
        for c in ALLY_COLORS:
            total += len(self.get_color_corners(c))
        return total


# ==============================
# 評估 & AI 選擇
# ==============================

def evaluate_move(state: GameState, move, color, alpha=0.2):
    new_state = state.apply_move(
        color,
        move["shape"],
        move["x"],
        move["y"],
        move["piece"]
    )
    cells_placed = len(move["shape"])
    ally_future_corners = new_state.count_ally_corners()
    score = cells_placed + alpha * ally_future_corners
    return score, new_state

def choose_best_move(state: GameState, color, alpha=0.2):
    moves = state.generate_legal_moves(color)
    if not moves:
        return None, None, None
    best_move = None
    best_score = float("-inf")
    best_state = None
    for m in moves:
        s, new_state = evaluate_move(state, m, color, alpha)
        if s > best_score:
            best_score = s
            best_move = m
            best_state = new_state
    return best_move, best_score, best_state


# ==============================
# CLI 互動主程式
# ==============================

def color_from_input(s: str):
    s = s.strip().upper()
    if s == "B":
        return BLUE
    if s == "Y":
        return YELLOW
    if s == "R":
        return RED
    if s == "G":
        return GREEN
    return None
def piece_ascii(piece_name, fill="██"):
    """
    用 BASE_PIECES 的 canonical shape 生成小圖（不管旋轉翻轉）
    """
    shape = normalize(BASE_PIECES[piece_name])
    max_x = max(x for x, y in shape)
    max_y = max(y for x, y in shape)

    lines = []
    for y in range(max_y + 1):
        row = []
        for x in range(max_x + 1):
            row.append(fill if (x, y) in shape else "  ")
        lines.append("".join(row))
    return lines

def print_remaining_pieces(state, color, per_row=6):
    """
    在棋盤下方印出該 color 剩餘棋子（形狀）
    """
    remain = sorted(state.remaining_pieces[color])
    if not remain:
        print("（已無剩餘棋子）")
        return

    blocks = []
    for name in remain:
        art = piece_ascii(name, fill="██")
        blocks.append((name, art))

    # 依 per_row 分行印出，並把不同高度的 piece 對齊
    i = 0
    while i < len(blocks):
        row_blocks = blocks[i:i+per_row]
        i += per_row

        # 每塊的高度不同，找本列最高
        max_h = max(len(art) for _, art in row_blocks)

        # 第一行：印名字
        name_line = "   ".join(f"{name:<8}" for name, _ in row_blocks)
        print(name_line)

        # 接著印 shape
        for y in range(max_h):
            parts = []
            for _, art in row_blocks:
                parts.append(art[y] if y < len(art) else " " * len(art[0]))
            print("   ".join(parts))
        print()
        
def shape_ascii_from_cells(shape_cells, fill="██"):
    shape = normalize(set(shape_cells))
    max_x = max(x for x, y in shape)
    max_y = max(y for x, y in shape)

    lines = []
    for y in range(max_y + 1):
        row = []
        for x in range(max_x + 1):
            row.append(fill if (x, y) in shape else "  ")
        lines.append("".join(row))
    return lines
def color_title(color, text):
    return COLOR_NAME[color] + " " + text
    
def main():
    state = GameState()
    print("=== Blokus 簡易 AI 助手（CLI 版）===")
    print("顏色代號：B=BLUE(我方)、Y=YELLOW(我方)、R=RED、G=GREEN")
    print("輸入 q 可離開\n")

    current_color = BLUE
    last_cells = set()
    while True:
        clear_screen()              # ⬅️ 新增：每次迴圈先清畫面
        print("=== Blokus 簡易 AI 助手（CLI 版）===")
        print("顏色代號：B=BLUE(我方)、Y=YELLOW(我方)、R=RED、G=GREEN")
        print("輸入 q 可離開\n")

        state.print_board(last_cells=last_cells)
        print()
        print("目前輪到顏色：", COLOR_NAME[current_color])
        print()
        #print_remaining_pieces(state, current_color, per_row=13)
        
        cmd = input(
            "輸入指令 [enter=AI 建議 / c=換顏色 / a=顯示全部剩餘棋子 / q=離開]："
        ).strip().lower()
        if cmd == "a":
            clear_screen()
            print("=== 所有顏色剩餘棋子一覽 ===\n")

            for color, name in [
                (BLUE, "BLUE (B)"),
                (YELLOW, "YELLOW (Y)"),
                (RED, "RED (R)"),
                (GREEN, "GREEN (G)"),
            ]:
                
                print(color_title(color, "剩餘棋子"))
                print_remaining_pieces(state, color, per_row=6)
                print()

            input("按 Enter 回到棋盤...")
            continue

        if cmd == "q":
            print("結束遊戲。")
            break
        if cmd == "c":
            c = input("請輸入顏色(B/Y/R/G)：")
            new_c = color_from_input(c)
            if new_c is None:
                print("顏色輸入錯誤。")
                continue
            current_color = new_c
            continue

        # 預設：AI 幫你選最佳一手
        move, score, new_state = choose_best_move(state, current_color, alpha=0.2)
        if move is None:
            print("這個顏色已沒有合法步可下。")
            # 簡單輪到下一個顏色（你之後可以改成只輪 B/Y）
            if current_color == BLUE:
                current_color = YELLOW
            elif current_color == YELLOW:
                current_color = RED
            elif current_color == RED:
                current_color = GREEN
            else:
                current_color = BLUE
            continue

        # AI 建議會覆蓋的座標（絕對座標）
        placed_cells = {(move["x"] + dx, move["y"] + dy) for (dx, dy) in move["shape"]}
        print(f"{FG_WHITE}\033[40m AI 建議 {RESET}")
        print(f"- 棋子：{move['piece']}")
        print(f"- 放置原點：(x={move['x']}, y={move['y']})")
        print(f"- 會覆蓋座標：{sorted(list(placed_cells))}")
        print(f"- 評分：{score:.2f}")

        # 印出 AI 建議的棋子形狀（依此 orientation）
        print("- 建議棋子形狀（此方向）：")
        for line in shape_ascii_from_cells(move["shape"], fill="██"):
            print("  " + line)
        print()
        
        # 這裡改成：直接按 Enter = 視為 y
        confirm = input("是否套用這一步？(y/n，直接 Enter = y)：").strip().lower()
        if confirm in ("", "y"):
            state = new_state
            last_cells = placed_cells  # ✅ 更新：下一次顯示時高亮這塊
            # 簡單輪到下一個顏色（你也可以改成只在 B/Y 間切換）
            if current_color == BLUE:
                current_color = YELLOW
            elif current_color == YELLOW:
                current_color = RED
            elif current_color == RED:
                current_color = GREEN
            else:
                current_color = BLUE
        else:
            print("未套用，您可以換顏色或再次請 AI 建議。")

if __name__ == "__main__":
    main()
