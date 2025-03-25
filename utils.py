import copy
from itertools import combinations, product
from collections import deque
import re
import os

def generate_all_possible_next_states(engine, mode='detector'):
    """
    현재 engine 상태에서 도달 가능한 다음 engine 상태들을 반환한다.
    이 함수는 phase에 따라 다르게 동작한다.

    return: List[Engine]  # 상태 복사본 목록
    """
    next_states = []

    phase = engine.phase

    if phase == 'char_selection':
        for char_name in engine.opened_characters:
            if char_name in engine.already_used_characters:
                continue
            new_engine = copy.deepcopy(engine)
            new_engine.already_used_characters = new_engine.already_used_characters + [char_name]
            
            # Holmes, Watson: 무조건 move_then_power
            if char_name in ['Holmes', 'Watson']:
                possible_orders = ['move_then_power']
            else:
                possible_orders = ['move_then_power', 'power_then_move']

            for order in possible_orders:
                new_engine.selected_character = char_name
                new_engine.selected_char_index = new_engine.character_names.index(char_name)
                new_engine.action_order = order
                new_engine.phase = 'move' if order == 'move_then_power' else 'power'
                next_states.append(new_engine)

    elif phase in ['move', 'power']:
        char_idx = engine.selected_char_index
        character = engine.characters[char_idx]

        if phase == 'move':
            move_coords = engine.valid_move(character, mode)
            for move in move_coords:
                new_engine = copy.deepcopy(engine)
                new_engine.characters[char_idx].coord = move

                # === 종료 조건 체크: move 직후 ===
                new_engine.is_end(character.name, move)

                # phase 변경
                if engine.action_order == 'move_then_power' and character.name != 'Gull':
                    new_engine.phase = 'power'
                else:
                    new_engine.phase = 'done'  # 후처리 시 필요
                next_states.append(new_engine)

        elif phase == 'power':
            power_candidates = generate_power_candidates(character, engine)
            for power_param in power_candidates:
                new_engine = copy.deepcopy(engine)
                character_copy = new_engine.characters[char_idx]
                if character_copy.name == 'Gull':
                    power_param = {'switch_character': next(c for c in new_engine.characters if c.name == power_param)}
                if character_copy.name == 'Goodley' and 'grab_character' in power_param:
                    name_list = power_param['grab_character']
                    char_objs = [c for c in new_engine.characters if c.name in name_list]
                    power_param = {
                        'grab_character': char_objs,
                        'target_coord': power_param['target_coord']
                    }
                if character_copy.name == 'Holmes':
                    if new_engine.round % 2:
                        if new_engine.action_index in [0, 3]:
                            current_turn = 'detector'
                        else:
                            current_turn = 'jack'
                    else:
                        if new_engine.action_index in [1, 2]:
                            current_turn = 'detector'
                        else:
                            current_turn = 'jack'
                    power_param = {
                        'current_turn': current_turn,
                    }
                character_copy.power(new_engine.board, power_param)

                if engine.action_order == 'power_then_move' and character.name != 'Gull':
                    new_engine.phase = 'move'
                else:
                    new_engine.phase = 'done'
                next_states.append(new_engine)

    elif phase == 'done':
        # 한 캐릭터의 행동이 모두 끝난 후 → 다음 캐릭터로 넘어가는 로직은 engine 쪽에서 처리
        pass

    return next_states

def generate_power_candidates(character, engine):
    name = character.name
    board = engine.board

    if name == 'Smith':
        # 켜진 조명 중 하나 끄고, 꺼진 조명 중 하나 켜기
        turn_on_list = board.extinguished_light
        turn_off_list = board.valid_light
        return [
            {'turn_on': on, 'turn_off': off}
            for on in turn_on_list
            for off in turn_off_list
            if on != off
        ]

    if name == 'Bert':
        # 닫힌 구멍 열고, 열린 구멍 닫기
        open_list = board.closed_hole
        close_list = board.valid_hole
        return [
            {'open_hole': open_hole, 'close_hole': close_hole}
            for open_hole in open_list
            for close_hole in close_list
            if open_hole != close_hole
        ]

    if name == 'Lestrade':
        # 닫힌 출구 열고, 열린 출구 닫기
        open_list = board.closed_exit
        close_list = board.valid_exit
        return [
            {'open_exit': open_exit, 'close_exit': close_exit}
            for open_exit in open_list
            for close_exit in close_list
            if open_exit != close_exit
        ]

    if name == 'Gull':
        # 다른 캐릭터들과 스왑 (자기 자신 제외)
        return ['Holmes', 'Watson', 'Smith', 'Lestrade', 'Stealthy', 'Goodley', 'Bert']

    if name == 'Watson':
        return [
            {'lantern_dir': dir} for dir in [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        ]

    if name == 'Goodley':
        goodley_coord = character.coord
        all_characters = engine.characters
        other_chars = [c for c in all_characters if c.name != 'Goodley']
        occupied = {c.coord for c in all_characters}

        # 모든 캐릭터에 대해 BFS 수행 한 번만
        shortest_paths_map = {}

        for c in other_chars:
            paths = bfs_all_shortest_paths(c.coord, goodley_coord, board)
            if paths:
                shortest_paths_map[c.name] = paths

        results = []

        # === 1명 선택 (3칸 당김) ===
        for name, paths in shortest_paths_map.items():
            three_step_targets = set()
            for path in paths:
                if len(path) >= 4:
                    three_step_targets.add(path[3])  # 캐릭터 기준 3칸 전 위치
            for target in three_step_targets:
                if target in occupied:
                    continue
                results.append({
                    'grab_character': [name],
                    'target_coord': [target]
                })

        # === 3명 선택 (1칸 당김) ===
        for name_triplet in combinations(shortest_paths_map.keys(), 3):
            coord_lists = []
            for name in name_triplet:
                one_step_targets = set()
                for path in shortest_paths_map[name]:
                    if len(path) >= 2:
                        one_step_targets.add(path[1])  # 캐릭터 기준 1칸 전
                if not one_step_targets:
                    break
                coord_lists.append(list(one_step_targets))

            if len(coord_lists) < 3:
                continue

            for coord_combo in product(*coord_lists):
                coord_set = set(coord_combo)
                if len(coord_set) < 3:
                    continue
                if goodley_coord in coord_set:
                    continue
                if any(c in occupied for c in coord_set):
                    continue

                results.append({
                    'grab_character': list(name_triplet),
                    'target_coord': list(coord_combo)
                })

        return results

    return [{}]

def bfs_all_shortest_paths(start, goal, board):
    directions = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    queue = deque([(start, [start])])
    visited = {start: 0}
    shortest_paths = []
    min_dist = None

    while queue:
        (y, x), path = queue.popleft()

        if (y, x) == goal:
            if min_dist is None:
                min_dist = len(path)
            if len(path) == min_dist:
                shortest_paths.append(path)
            continue

        if min_dist is not None and len(path) > min_dist:
            continue  # 이미 최단거리보다 길면 skip

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < board.grid_size[0] and 0 <= nx < board.grid_size[1]:
                if board.grid[ny][nx] in [1, 3]:
                    next_coord = (ny, nx)
                    if next_coord not in visited or visited[next_coord] >= len(path) + 1:
                        visited[next_coord] = len(path) + 1
                        queue.append((next_coord, path + [next_coord]))

    return shortest_paths  # [ [start,...,goal], ... ]

def mk_experiment_dir(base_dir="runs", prefix="jack_detector_ppo"):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    numbers = []

    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if match:
            numbers.append(int(match.group(1)))

    next_num = max(numbers, default=0) + 1
    new_dir_name = f"{prefix}_{next_num}"
    new_dir_path = os.path.join(base_dir, new_dir_name)

    # 디렉토리 생성
    os.makedirs(new_dir_path)
    os.makedirs(os.path.join(new_dir_path, 'ckpt/'))

    return new_dir_path

if __name__ == '__main__':
    from env import Engine
    import copy

    engine = Engine()

    # 테스트용 설정: Goodley의 power phase
    engine.phase = 'power'
    engine.selected_character = 'Holmes'
    engine.selected_char_index = engine.character_names.index('Holmes')
    engine.action_order = 'move_then_power'  # power 먼저
    engine.opened_characters = ['Holmes']
    engine.already_used_characters = []

    print("✅ Goodley power phase 상태에서 가능한 next states 생성 중...")
    next_states = generate_all_possible_next_states(engine)
    print(f"생성된 next states 수: {len(next_states)}")

    # 최대 5개까지 시각화
    for i, new_engine in enumerate(next_states[:5]):
        print(f"\n==== 후보 상태 {i+1} ====")
        print(new_engine.board.evidence_deck)
        new_engine.visualize_board()