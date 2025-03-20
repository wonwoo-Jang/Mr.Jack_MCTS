# import numpy as np
import random

class Character:
    def __init__(self, name, max_move, start_coord, is_jack):
        self.max_move = max_move
        self.name = name
        if name == 'Watson':
            # 방향 기준: x, y방향으로 좌표가 1, 1 증가하는 방향이 1, 그걸 기준으로 반시계 방향으로 증가
            self.lantern_dir = 2
        self.coord = start_coord
        self.is_jack = is_jack
        self.jack_watched = True
    
    def move(self, new_coord):
        self.coord = new_coord
    
    def power(self, board, extra_params):
        if self.name == 'Homes':
            ## 증거 카드 반환
            evidence = random.choice(board.evidence_deck)
            board.evidence_deck.remove(evidence)
            
            return evidence

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
        self.valid_light = [(12, 1), (4, 11), (3, 2), (13, 10), (10, 5), (6, 7)]
        self.extinguished_light = [(2, 5), (14, 7)]
        self.valid_exit = [(1, 0), (15, 12)]
        self.closed_exit = [(16, 1), (0, 11)]
        self.evidence_deck = ['Homes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
        self.action_deck = ['Homes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
    
class Engine:
    def __init__(self):
        self.board = Board()
        
        # Game property
        self.round = 1
        self.max_round = 8
        self.isEnd = False
        self.jack_watched = True

        # pick Mr.jack from evidence deck
        jack = random.choice(self.board.evidence_deck)
        self.board.evidence_deck.remove(jack)
        
        self.character_names = ['Homes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
        self.characters = [Character('Homes', 3, (11, 6), jack=='Homes'),
                           Character('Watson', 3, (15, 8), jack=='Watson'),
                           Character('Smith', 3, (5, 6), jack=='Smith'),
                           Character('Lestrade', 3, (9, 4), jack=='Lestrade'),
                           Character('Stealthy', 4, (9, 0), jack=='Stealthy'),
                           Character('Goodley', 3, (7, 12), jack=='Goodley'),
                           Character('Gull', 3, (1, 4), jack=='Gull'),
                           Character('Bert', 3, (7, 8), jack=='Bert'),]
        
    def valid_move(self, target_character):
        from collections import deque

        start_y, start_x = target_character.coord
        queue = deque([(start_y, start_x, 0)])  # (y, x, 이동 횟수)
        visited = set([(start_y, start_x)])  # 중복 이동 방지

        directions = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 형식이 y, x임에 주의!

        possible_moves = set()  # 이동 가능한 좌표 저장

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
                            
                            if self.board.grid[ny][nx] in (1, 3, 4):
                                possible_moves.add((ny, nx))
                    else:
                        if self.board.grid[ny][nx] in (1, 3, 4) and (ny, nx) not in visited:
                            possible_moves.add((ny, nx))
                            queue.append((ny, nx, moves + 1))
                            visited.add((ny, nx))

            # 구멍(3)에서 다른 구멍(3)으로 이동 가능
            if self.board.grid[cy][cx] == 3:
                for hy, hx in self.board.valid_hole:
                    if (hy, hx) != (cy, cx) and (hy, hx) not in visited:
                        possible_moves.add((hy, hx))
                        queue.append((hy, hx, moves + 1))
                        visited.add((hy, hx))

        return list(possible_moves)

    def simulate(self, detector_agent, jack_agent):
        while self.round <= self.max_round:
            # character action phase
            if self.round % 2 == 1:
                random.shuffle(self.board.action_deck)
                opened_character = self.board.action_deck[:4]
                self.board.action_deck = self.board.action_deck[4:]
                
                # detector select 1
                selected_character = detector_agent.search_select_character(self, opened_character, next_turn=False)
                opened_character.remove(selected_character)
                detector_agent.search_action(self, selected_character)
                
                # jack select 2
                selected_character = jack_agent.search_select_character(self, opened_character, next_turn=True)
                opened_character.remove(selected_character)
                jack_agent.search_action(self, selected_character)
                selected_character = jack_agent.search_select_character(self, opened_character, next_turn=False)
                opened_character.remove(selected_character)
                jack_agent.search_action(self, selected_character)
                
                # detector select 1
                selected_character = detector_agent.search_select_character(self, opened_character, next_turn=False)
                opened_character.remove(selected_character)
                detector_agent.search_action(self, selected_character)
                
            else:
                opened_character = self.board.action_deck
                
                # jack select 1
                selected_character = jack_agent.search_select_character(self, opened_character, next_turn=False)
                opened_character.remove(selected_character)
                jack_agent.search_action(self, selected_character)
                
                # detector select 2
                selected_character = detector_agent.search_select_character(self, opened_character, next_turn=True)
                opened_character.remove(selected_character)
                detector_agent.search_action(self, selected_character)
                selected_character = detector_agent.search_select_character(self, opened_character, next_turn=False)
                opened_character.remove(selected_character)
                detector_agent.search_action(self, selected_character)
                
                # jack select 1
                selected_character = jack_agent.search_select_character(self, opened_character, next_turn=False)
                opened_character.remove(selected_character)
                jack_agent.search_action(self, selected_character)
                
                self.board.action_deck = ['Homes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Gull', 'Bert']
                
            # observer check
            jack_index = self.character_names.index(self.jack)
            #NOTE: 목격자 확인하는 함수 구현하기
            
            # turn off the light
            if self.round <= 4:
                target_light = self.board.valid_light[0]
                self.board.valid_light = self.board.valid_light[1:]
                self.board.extinguished_light.append(target_light)
            
            round += 1

    def test(self):
        for xy in self.board.valid_hole:
            print(self.board.grid[xy[0]][xy[1]])
        self.characters[2].power(self.board, {'turn_on': (2, 5), 'turn_off': (3, 2)})
        print(self.board.valid_light, self.board.extinguished_light)
        self.characters[3].power(self.board, {'close_exit': (1, 0), 'open_exit': (16, 1)})
        print(self.board.valid_exit, self.board.closed_exit)
        self.characters[7].power(self.board, {'close_hole': (4, 1), 'open_hole': (14, 1)})
        print(self.board.valid_hole, self.board.closed_hole)
        # print(self.characters[6].coord, self.characters[3].coord)
        # self.characters[6].power(self.board, {'switch_character': self.characters[2]})
        # print(self.characters[6].coord, self.characters[3].coord)
        evidence = self.characters[0].power(self.board, None)
        print(evidence, self.board.evidence_deck)
        
        print(self.characters[4].coord, self.valid_move(self.characters[4]))

engine = Engine()
engine.test()