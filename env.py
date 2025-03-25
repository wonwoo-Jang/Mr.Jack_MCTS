import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Wedge
from collections import deque
import random
import numpy as np
import torch

class Character:
    def __init__(self, name, max_move, start_coord, is_jack):
        self.max_move = max_move
        self.name = name
        if name == 'Watson':
            self.lantern_dir = (-1, 1)
        self.coord = start_coord
        self.is_jack = is_jack
        self.jack_watched = True
    
    def move(self, new_coord):
        self.coord = new_coord
    
    def power(self, board, extra_params):
        if self.name == 'Holmes':
            evidence = random.choice(board.evidence_deck)
            board.evidence_deck.remove(evidence)
            if extra_params['current_turn'] == 'detector':
                board.detector_evidence.append(evidence)
            else:
                board.jack_evidence.append(evidence)
            
            return

        if self.name == 'Watson':
            self.lantern_dir = extra_params['lantern_dir']

            return

        if self.name == 'Smith':
            turn_on = extra_params['turn_on']
            turn_off = extra_params['turn_off']
            off_light = board.valid_light.index(turn_off)
            on_light = board.extinguished_light.index(turn_on)
            board.valid_light[off_light] = turn_on
            board.extinguished_light[on_light] = turn_off
            
            return

        if self.name == 'Lestrade':
            close_exit = extra_params['close_exit']
            open_exit = extra_params['open_exit']
            closed_index = board.valid_exit.index(close_exit)
            opened_index = board.closed_exit.index(open_exit)
            board.valid_exit[closed_index] = open_exit
            board.closed_exit[opened_index] = close_exit

            return

        if self.name == 'Stealthy':

            return
        
        if self.name == 'Goodley':
            target_character = extra_params['grab_character']
            target_coord = extra_params['target_coord']
            if len(target_character) == 1:
                target_character[0].coord = target_coord[0]
            else:
                for i in range(3):
                    target_character[i].coord = target_coord[i]

            return

        if self.name == 'Gull':
            target_character = extra_params['switch_character']
            prev_coord = self.coord
            self.coord = target_character.coord
            target_character.coord = prev_coord

            return

        if self.name == 'Bert':
            open_hole = extra_params['open_hole']
            close_hole = extra_params['close_hole']
            closed_index = board.valid_hole.index(close_hole)
            opened_index = board.closed_hole.index(open_hole)
            board.valid_hole[closed_index] = open_hole
            board.closed_hole[opened_index] = close_hole
            
            return

        return

class Board:
    def __init__(self):
        self.grid_size = (17, 13)
        self.grid = [
            # 0: wall, 1; valid, 2: light, 3: hole, 4: exit
            [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0],
            [4, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 3, 0, 1, 0, 3, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
            [0, 3, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 4],
            [0, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]
        ]
        self.valid_hole = [(0, 5), (4, 1), (7, 10), (9, 2), (9, 6), (12, 11), (16, 7)]
        self.closed_hole = [(2, 11), (14, 1)]
        self.hole = [(0, 5), (4, 1), (7, 10), (9, 2), (9, 6), (12, 11), (16, 7), (2, 11), (14, 1)]
        self.valid_light = [(12, 1), (4, 11), (3, 2), (13, 10), (10, 5), (6, 7)]
        self.extinguished_light = [(2, 5), (14, 7)]
        self.light = [(12, 1), (4, 11), (3, 2), (13, 10), (10, 5), (6, 7), (2, 5), (14, 7)]
        self.valid_exit = [(1, 0), (15, 12)]
        self.closed_exit = [(16, 1), (0, 11)]
        self.exit = [(1, 0), (15, 12), (16, 1), (0, 11)]
        self.evidence_deck = ['Holmes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
        self.detector_evidence = [] # detector가 뽑은 evidence character
        self.jack_evidence = []
        self.action_deck = ['Holmes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']

class Engine:
    def __init__(self):
        self.board = Board()
        # pick Mr.jack from evidence deck
        self.jack = random.choice(self.board.evidence_deck)
        self.board.evidence_deck.remove(self.jack)
        
        self.character_names = ['Holmes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
        self.characters = [Character('Holmes', 3, (11, 6), self.jack=='Holmes'),
                           Character('Watson', 3, (15, 8), self.jack=='Watson'),
                           Character('Smith', 3, (5, 6), self.jack=='Smith'),
                           Character('Lestrade', 3, (9, 4), self.jack=='Lestrade'),
                           Character('Stealthy', 4, (9, 0), self.jack=='Stealthy'),
                           Character('Goodley', 3, (7, 12), self.jack=='Goodley'),
                           Character('Gull', 3, (1, 4), self.jack=='Gull'),
                           Character('Bert', 3, (7, 8), self.jack=='Bert'),]

        # Game property
        self.end = False
        self.jack_watched = True
        self.jack_win = True
        self.excluded_jack_names = [] # observed 정보가 달라서 라운드가 끝나면 update

        self.round = 1
        self.max_round = 8
        self.action_index = 0       # 라운드 안에서 진행되는 작은 라운드, 0~3 값
        self.phase = "char_selection"  # char_selection, move, power, done (char_selection에서 move와 power순서도 정함)
        self.selected_character = None # char_name
        self.action_order = None # move_then_power, power_then_move
        random.shuffle(self.board.action_deck)
        self.opened_characters = self.board.action_deck[:4]
        self.board.action_deck = self.board.action_deck[4:]
        self.already_used_characters = []

    def get_observed(self):
        light_grid = np.zeros(self.board.grid_size)
        directions = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 형식이 y, x임에 주의!
        for character in self.characters:
            cy, cx = character.coord
            for dir in directions:
                dy, dx = dir
                if 0<=cy+dy<self.board.grid_size[0] and 0<=cx+dx<self.board.grid_size[1]:
                    light_grid[cy+dy][cx+dx] = 1
            
            if character.name == 'Watson':
                dy, dx = character.lantern_dir
                while 0<=cy+dy<self.board.grid_size[0] and 0<=cx+dx<self.board.grid_size[1] and self.board.grid[cy+dy][cx+dx] in [1, 3]:
                    light_grid[cy+dy][cx+dx] = 1
                    cy, cx = cy + dy, cx + dx
        
        for valid_light in self.board.valid_light:
            cy, cx = valid_light
            for dir in directions:
                dy, dx = dir
                light_grid[cy+dy][cx+dx] = 1

        return light_grid, [light_grid[character.coord[0]][character.coord[1]] for character in self.characters]

    def valid_move(self, target_character, mode='detector'):
        start_y, start_x = target_character.coord
        char_coords = [char.coord for char in self.characters]

        queue = deque([(start_y, start_x, 0)])
        visited = set([(start_y, start_x)])

        directions = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 형식이 y, x임에 주의!

        possible_moves = set()

        while queue:
            cy, cx, moves = queue.popleft()
            if moves >= target_character.max_move:
                continue

            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.board.grid_size[0] and 0 <= nx < self.board.grid_size[1]:
                    if target_character.name == 'Stealthy':
                        if (ny, nx) not in visited:
                            queue.append((ny, nx, moves + 1))
                            visited.add((ny, nx))
                            
                            if self.board.grid[ny][nx] in (1, 3):
                                if mode == 'jack' and (ny, nx) in char_coords:
                                    continue
                                possible_moves.add((ny, nx))
                            if mode == 'jack' and (ny, nx) in self.board.valid_exit and target_character.name == self.jack and not self.jack_watched:
                                possible_moves.add((ny, nx))
                    else:
                        if  (ny, nx) not in visited:
                            if self.board.grid[ny][nx] in (1, 3):
                                queue.append((ny, nx, moves + 1))
                                visited.add((ny, nx))
                                if mode == 'jack' and (ny, nx) in char_coords:
                                    continue
                                possible_moves.add((ny, nx))
                            if mode == 'jack' and (ny, nx) in self.board.valid_exit and target_character.name == self.jack and not self.jack_watched:
                                possible_moves.add((ny, nx))

            # move hole to hole
            if self.board.grid[cy][cx] == 3:
                for hy, hx in self.board.valid_hole:
                    if (hy, hx) != (cy, cx) and (hy, hx) not in visited:
                        queue.append((hy, hx, moves + 1))
                        visited.add((hy, hx))
                        if mode == 'jack' and (hy, hx) in char_coords:
                            continue
                        possible_moves.add((hy, hx))

        return list(possible_moves)

    def is_end(self, name, move_coord):
        # 1. capture
        for character in self.characters:
            if character.coord == move_coord and character.name != name:
                if character.name == self.jack:
                    self.jack_win = False
                else:
                    self.jack_win = True
                self.end = True

                return True

        # 2. escape
        for valid_exit in self.board.valid_exit:
            if valid_exit == move_coord:
                if name == self.jack and not self.jack_watched:
                    self.jack_win = True
                else:
                    self.jack_win = False
                self.end = True

                return True
            
        self.end = False

        return False

    def step(self):
        if self.phase == 'done':
            # 라운드가 끝난 경우
            if self.action_index == 3:
                # 8 라운드가 끝났으면 게임 종료
                if self.round >= self.max_round:
                    self.end = True
                    self.jack_win = True

                    return

                # 다음 라운드가 8라운드 이하면
                self.phase = 'char_selection'
                self.action_index = 0
                self.selected_character = None
                self.action_order = None
                if not self.round % 2:
                    self.board.action_deck = ['Holmes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
                    random.shuffle(self.board.action_deck)
                self.opened_characters = self.board.action_deck[:4]
                self.board.action_deck = self.board.action_deck[4:]
                self.already_used_characters = []

                # observer check
                jack_index = self.character_names.index(self.jack)
                _, observed_state = self.get_observed()
                self.jack_watched = (observed_state[jack_index] == 1)

                # exclude jack
                new_excluded = []

                for name, is_observed in zip(self.character_names, observed_state):
                    if self.jack_watched and not is_observed:
                        new_excluded.append(name)
                    elif not self.jack_watched and is_observed:
                        new_excluded.append(name)

                self.excluded_jack_names = list(set(self.excluded_jack_names + new_excluded))

                # turn off the light
                if self.round <= 4:
                    target_light = self.board.valid_light[0]
                    self.board.valid_light = self.board.valid_light[1:]
                    self.board.extinguished_light.append(target_light)

                self.round += 1

            # 라운드가 안 끝난 경우
            else:
                self.phase = 'char_selection'
                self.action_index += 1
                self.selected_character = None
                self.action_order = None

        return

    def visualize_board(self):
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_aspect('equal')
        ax.axis('off')

        hex_radius = 0.04
        dx = 3/2 * hex_radius
        dy = np.sqrt(3) * hex_radius

        color_map = {
            1: 'white', 2: 'gold', 3: 'white', 4: 'dodgerblue'
        }

        for row in range(self.board.grid_size[0]):
            for col in range(self.board.grid_size[1]):
                value = self.board.grid[row][col]
                if not value:
                    continue

                coord = (row, col)
                color = color_map.get(value, 'gray')
                if coord in self.board.extinguished_light:
                    color = 'darkgoldenrod'

                x = col * dx
                y = row//2 * dy + (dy/2 if not (col % 2) else 0)

                hexagon = RegularPolygon(
                                            xy=(x+0.2, -y+0.7),
                                            numVertices=6,
                                            radius=hex_radius,
                                            orientation=np.radians(30),
                                            facecolor=color,
                                            edgecolor='black',
                                            linewidth=1
                                        )
                ax.add_patch(hexagon)

                if value == 3:
                    ax.add_patch(Circle((x+0.2, -y+0.7), hex_radius * 0.3, color='black'))

                if coord in self.board.closed_hole or coord in self.board.closed_exit:
                    ax.text(x+0.2, -y+0.7, 'X', ha='center', va='center',
                            fontsize=10, color='red', fontweight='bold')

        for character in self.characters:
            cy, cx = character.coord
            x = cx * dx
            y = cy//2 * dy + (dy / 2 if not (cx % 2) else 0)

            name_color = 'blue' if self.selected_character == character.name else 'red'

            ax.text(x+0.2, -y+0.7, character.name[:2], ha='center', va='center',
                    fontsize=12, color=name_color, weight='bold',
                    bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec=name_color, lw=2))

            if character.name == 'Watson':
                dir_map = {(-2, 0): 90, (-1, 1): 30, (1, 1): 330,
                        (2, 0): 270, (1, -1): 210, (-1, -1): 150}
                if character.lantern_dir in dir_map:
                    angle = dir_map[character.lantern_dir]
                    ax.add_patch(Wedge((x+0.2, -y+0.7), hex_radius,
                                    angle - 30, angle + 30,
                                    facecolor='yellow', alpha=0.6))

        # === 추가 정보 표시 영역 ===
        info_x = 1.0
        info_y = 0.8
        gap = 0.05

        ax.text(info_x, info_y, f"Phase: {self.phase}", color='white', fontsize=12, transform=ax.transAxes)
        info_y -= gap
        if self.selected_character:
            ax.text(info_x, info_y, f"Selected: {self.selected_character}", color='skyblue', fontsize=12, transform=ax.transAxes)
            info_y -= gap

        if self.round % 2:
            if self.action_index in [0, 3]:
                current_turn = 'detector'
            else:
                current_turn = 'jack'
        else:
            if self.action_index in [1, 2]:
                current_turn = 'detector'
            else:
                current_turn = 'jack'
        ax.text(info_x, info_y, f"Current Turn: {current_turn}", color='lightgreen', fontsize=12, transform=ax.transAxes)
        info_y -= gap

        ax.text(info_x, info_y, f"Round: {self.round} / {self.max_round}", color='orange', fontsize=12, transform=ax.transAxes)
        info_y -= gap
        ax.text(info_x, info_y, f"Sub-Round: {self.action_index + 1} / 4", color='orange', fontsize=12, transform=ax.transAxes)
        info_y -= gap * 2

        ax.text(info_x, info_y, f"Mr. jack: {self.jack}", color='magenta', fontsize=12, transform=ax.transAxes)
        info_y -= gap

        ax.text(info_x, info_y, "jack Candidates:", color='white', fontsize=12, transform=ax.transAxes)
        info_y -= gap
        for name in self.character_names:
            mark = " (X)" if name in set(self.excluded_jack_names + self.board.detector_evidence) else ""
            ax.text(info_x + 0.02, info_y, f"- {name}{mark}", color='white', fontsize=11, transform=ax.transAxes)
            info_y -= gap * 0.8

        if self.end:
            winner = "jack" if self.jack_win else "detector"
            ax.text(info_x, info_y, f"{winner} wins!", color='cyan', fontsize=14, fontweight='bold', transform=ax.transAxes)
            info_y -= gap

        circle_x, circle_y = 0.5, -0.08  # Axes 좌표 기준 위치
        ax.text(0.5, -0.03, "Jack Observed State", fontsize=12, color='white',
                ha='center', transform=ax.transAxes)

        jack_circle_color = 'white' if self.jack_watched else 'gray'
        circle = Circle((circle_x, circle_y), 0.02, color=jack_circle_color,
                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(circle)

        plt.tight_layout()
        plt.show()

    def extract_state(self, mode='detector'):
        # (1) 보드 입력 구성
        board = np.array(self.board.grid, dtype=np.float32)
        board_overlay = board.copy()

        # 보드 오버레이: 조명, 구멍, 출구 등 추가
        for coords, val in [
            (self.board.valid_light, 2),
            (self.board.extinguished_light, 3),
            (self.board.valid_exit, 4),
            (self.board.closed_exit, 5),
            (self.board.valid_hole, 6),
            (self.board.closed_hole, 7),
        ]:
            coords = np.array(coords)
            if coords.size > 0:
                board_overlay[coords[:, 0], coords[:, 1]] = val

        # 캐릭터 위치 맵
        char_map = np.zeros_like(board, dtype=np.float32)
        coords = np.array([char.coord for char in self.characters])
        indices = np.arange(1, len(self.characters) + 1, dtype=np.float32)
        char_map[coords[:, 0], coords[:, 1]] = indices

        # 목격 정보 맵
        observed_map, _ = self.get_observed()

        # board_input: (1, 3, 17, 13)
        board_input = np.stack([board_overlay, char_map, observed_map]).astype(np.float32)
        board_input = torch.tensor(board_input).unsqueeze(0)

        # (2) 벡터 입력 구성 (misc_input)
        char_names = self.character_names

        # jack 후보 마스크 (8,): detector면 evidence에 없는 카드, jack이면 jack이 얻은 evidence가 input
        if mode == 'detector':
            jack_candidates = [candidates not in self.board.detector_evidence for candidates in self.character_names]
            jack_input = torch.tensor([
                1.0 if (name in jack_candidates and name not in self.excluded_jack_names) else 0.0
                for name in self.character_names
            ], dtype=torch.float32)
        else:
            jack_input = torch.tensor([
                1.0 if (name in self.board.jack_evidence) else 0.0
                for name in self.character_names
            ], dtype=torch.float32)

        if mode == 'jack':
            # 실제 jack 정보
            real_jack_input = torch.tensor([
                1.0 if name == self.jack else 0.0
                for name in self.character_names
            ], dtype=torch.float32)

        # 오픈된 캐릭터 마스크 (8,)
        opened_mask = torch.tensor(
            [1.0 if name in self.opened_characters else 0.0 for name in char_names],
            dtype=torch.float32
        )

        # 사용 가능한 캐릭터 마스크 (8,)
        available_mask = torch.tensor(
            [1.0 if name in self.opened_characters and name not in self.already_used_characters else 0.0 for name in char_names],
            dtype=torch.float32
        )

        # round, action_index (1,)
        round_info = torch.tensor([self.round / self.max_round], dtype=torch.float32)
        action_phase_info = torch.tensor([self.action_index / 4], dtype=torch.float32)

        # phase flags (3,)
        phase_flags = {
            'char_selection': [1, 0, 0],
            'move': [0, 1, 0],
            'power': [0, 0, 1],
        }
        phase_flag_tensor = torch.tensor(phase_flags.get(self.phase, [0, 0, 0]), dtype=torch.float32)

        # 행동 순서 정보 ('move_then_power': 0.0, 'power_then_move': 1.0)
        if self.action_order == 'move_then_power':
            order_tensor = torch.tensor([0.0], dtype=torch.float32)
        elif self.action_order == 'power_then_move':
            order_tensor = torch.tensor([1.0], dtype=torch.float32)
        else:
            order_tensor = torch.tensor([-1.0], dtype=torch.float32)  # 선택 전 상태

        # 선택된 캐릭터 one-hot (없으면 all-zero)
        if self.selected_character is None:
            selected_char_tensor = torch.zeros(8)
        else:
            selected_idx = char_names.index(self.selected_character)
            selected_char_tensor = torch.nn.functional.one_hot(torch.tensor(selected_idx), num_classes=8).float()

        # 남은 증거 카드 수 (정규화)
        evidence_count = torch.tensor([len(self.board.evidence_deck) / 8.0], dtype=torch.float32)

        # jack이 목격 상태인지
        jack_watched = torch.tensor([float(self.jack_watched)], dtype=torch.float32)

        # (3) 최종 misc_input
        misc_input = torch.cat([
            jack_input,             # (8,)
            opened_mask,            # (8,)
            available_mask,         # (8,)
            round_info,             # (1,)
            action_phase_info,      # (1,)
            phase_flag_tensor,      # (3,)
            order_tensor,           # (1,)
            selected_char_tensor,   # (8,)
            evidence_count,         # (1,)
            jack_watched            # (1,)
        ]).unsqueeze(0)             # 최종 shape: (1, 40)

        if mode == 'jack':
            misc_input = torch.cat([
                jack_input,             # (8,)
                real_jack_input,        # (8,)
                opened_mask,            # (8,)
                available_mask,         # (8,)
                round_info,             # (1,)
                action_phase_info,      # (1,)
                phase_flag_tensor,      # (3,)
                order_tensor,           # (1,)
                selected_char_tensor,   # (8,)
                evidence_count,         # (1,)
                jack_watched            # (1,)
            ]).unsqueeze(0)             # 최종 shape: (1, 48)

        return board_input, misc_input

if __name__ == '__main__':
    engine = Engine()
    engine.visualize_board()