import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class CharacterOrderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        self.char_head = nn.Linear(256, 8)
        self.order_head = nn.Linear(256, 2)

    def forward(self, board_input, misc_input):
        misc_exp = misc_input.unsqueeze(-1).unsqueeze(-1)
        misc_map = misc_exp.expand(-1, -1, 17, 13)
        x = torch.cat([board_input, misc_map], dim=1)
        x = self.cnn(x)
        x = self.fc(x)
        return self.char_head(x), self.order_head(x)

class MoveSelectionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim + 8, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        self.move_head = nn.Linear(256, 17 * 13)

    def forward(self, board_input, misc_input, selected_char_idx):
        misc_exp = misc_input.unsqueeze(-1).unsqueeze(-1)
        misc_map = misc_exp.expand(-1, -1, 17, 13)

        # 캐릭터 one-hot을 spatial map으로 변환 (벡터화)
        batch_size = board_input.size(0)
        onehot_map = torch.zeros((batch_size, 8, 1, 1), device=board_input.device)
        onehot_map.scatter_(1, selected_char_idx.view(-1, 1, 1, 1), 1.0)
        onehot_map = onehot_map.expand(-1, -1, 17, 13)  # (B, 8, 17, 13)

        x = torch.cat([board_input, misc_map, onehot_map], dim=1)
        x = self.cnn(x)
        x = self.fc(x)
        return self.move_head(x).view(-1, 17, 13)

class PowerSelectionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        self.power_heads = nn.ModuleDict({
            'Watson': nn.Linear(256, 6),
            'Smith': nn.Linear(256, 8 * 7),
            'Lestrade': nn.Linear(256, 4 * 3),
            'Goodley': nn.Linear(256, 2 + 8 * 3 * 2),
            'Bert': nn.Linear(256, 9 * 8),
        })

    def forward(self, board_input, misc_input, character_name):
        misc_exp = misc_input.unsqueeze(-1).unsqueeze(-1)
        misc_map = misc_exp.expand(-1, -1, 17, 13)
        x = torch.cat([board_input, misc_map], dim=1)
        x = self.cnn(x)
        x = self.fc(x)
        return self.power_heads[character_name](x)

class Detector_agent:
    def __init__(self):
        self.jack_candidates = np.ones(8)
        self.input_dim = 21 # board_input: (B, 3, 17, 13), misc_input: (B, 18), 18 + 3=21
        self.hidden_dim = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.character_and_order_net = CharacterOrderNet(self.input_dim, self.hidden_dim).to(self.device)
        self.move_net = MoveSelectionNet(self.input_dim, self.hidden_dim).to(self.device)
        self.power_net = PowerSelectionNet(self.input_dim, self.hidden_dim).to(self.device)

    def extract_state(self, engine, opened_characters, next_turn):
        board = np.array(engine.board.grid, dtype=np.float32)  # (17, 13)
        board_overlay = board.copy()

        # board 상태 표시
        for coords, val in [
            (engine.board.valid_light, 2),
            (engine.board.extinguished_light, 3),
            (engine.board.valid_exit, 4),
            (engine.board.closed_exit, 5),
            (engine.board.valid_hole, 6),
            (engine.board.closed_hole, 7),
        ]:
            coords = np.array(coords)
            if coords.size > 0:
                board_overlay[coords[:, 0], coords[:, 1]] = val

        # 캐릭터 위치 맵 만들기 (0: 없음, 1~8: 캐릭터 번호)
        char_map = np.zeros_like(board, dtype=np.float32)
        coords = np.array([char.coord for char in engine.characters])
        indices = np.arange(1, len(engine.characters) + 1, dtype=np.float32)
        char_map[coords[:, 0], coords[:, 1]] = indices

        # 목격 정보 맵
        observed_map, _ = engine.get_observed()

        # CNN 입력 (3, 17, 13)
        board_input = np.stack([board_overlay, char_map, observed_map]).astype(np.float32)
        board_input = torch.tensor(board_input).unsqueeze(0)  # (1, 3, 17, 13)

        # 기타 입력: jack 후보 (8,) + action 카드 (8,) + next_turn (1,) + round (1,) → 총 (18,)
        jack_input = torch.tensor(self.jack_candidates, dtype=torch.float32)
        char_names = engine.character_names
        action_mask = torch.tensor([1.0 if name in opened_characters else 0.0 for name in char_names])
        next_turn = torch.tensor([next_turn], dtype=torch.float32)
        round_info = torch.tensor([engine.round / engine.max_round])
        misc_input = torch.cat([jack_input, action_mask, next_turn, round_info]).unsqueeze(0)  # (1, 18)

        return board_input, misc_input

    def mask_smith_power_logits(self, logits, engine):
      light_positions_np = np.array(engine.board.light)
      valid_light_np = np.array(engine.board.valid_light)
      extinguished_np = np.array(engine.board.extinguished_light)

      # 전체 조합 (56, 2)
      smith_action_space = np.array([
          (i, j) for i in range(8) for j in range(8) if i != j
      ])

      # 각각 인덱스에 해당하는 위치값 얻기
      src_pos = light_positions_np[smith_action_space[:, 0]]  # (56, 2)
      tgt_pos = light_positions_np[smith_action_space[:, 1]]  # (56, 2)

      # (56,) → 유효한 조합: src ∈ valid AND tgt ∈ extinguished
      valid_mask = (
          (src_pos[:, None] == valid_light_np).all(-1).any(-1) &
          (tgt_pos[:, None] == extinguished_np).all(-1).any(-1)
      )

      # 마스킹
      masked_logits = logits.clone()
      masked_logits[~torch.tensor(valid_mask, device=logits.device)] = float('-inf')

      return masked_logits, smith_action_space

    def mask_bert_power_logits(self, logits, engine):
        hole_positions_np = np.array(engine.board.hole)
        valid_hole_np = np.array(engine.board.valid_hole)
        closed_hole_np = np.array(engine.board.closed_hole)

        bert_action_space = np.array([(i, j) for i in range(9) for j in range(9) if i != j])

        src_pos = hole_positions_np[bert_action_space[:, 0]]
        tgt_pos = hole_positions_np[bert_action_space[:, 1]]

        valid_mask = (
            (src_pos[:, None] == valid_hole_np).all(-1).any(-1) &
            (tgt_pos[:, None] == closed_hole_np).all(-1).any(-1)
        )

        masked_logits = logits.clone()
        masked_logits[~torch.tensor(valid_mask, device=logits.device)] = float('-inf')

        return masked_logits, bert_action_space

    def mask_lestrade_power_logits(self, logits, engine):
        valid_np = np.array(engine.board.valid_exit)
        closed_np = np.array(engine.board.closed_exit)
        lestrade_action_space = np.array([(i, j) for i in range(4) for j in range(4) if i != j])

        src_pos = np.array([engine.board.exit[i] for i, _ in lestrade_action_space])
        tgt_pos = np.array([engine.board.exit[j] for _, j in lestrade_action_space])

        valid_mask = (
            (src_pos[:, None] == valid_np).all(-1).any(-1) &
            (tgt_pos[:, None] == closed_np).all(-1).any(-1)
        )

        masked_logits = logits.clone()
        masked_logits[~torch.tensor(valid_mask, device=logits.device)] = float('-inf')
        return masked_logits, lestrade_action_space

    def search_character_and_order(self, engine, opened_characters, next_turn):
      board_input, misc_input = self.extract_state(engine, opened_characters, next_turn)
      board_input = board_input.to(self.device)
      misc_input = misc_input.to(self.device)
      char_logits, order_logits = self.character_and_order_net(board_input, misc_input)
      char_logits = char_logits[0]
      char_mask = torch.tensor([0.0 if name in opened_characters else float('-inf') for name in engine.character_names], device=char_logits.device)
      selected_char_idx = torch.argmax(char_logits + char_mask).item()
      selected_char_name = engine.character_names[selected_char_idx]

      if selected_char_name in ['Holmes', 'Watson', 'Stealthy']:
          order = 'move_then_power'
      else:
          order = 'power_then_move' if torch.argmax(order_logits[0]).item() == 0 else 'move_then_power'

      return selected_char_idx, selected_char_name, order

    def search_move(self, engine, opened_characters, next_turn, selected_char_idx):
        board_input, misc_input = self.extract_state(engine, opened_characters, next_turn)
        board_input = board_input.to(self.device)
        misc_input = misc_input.to(self.device)
        selected_char_idx_tensor =  torch.tensor([selected_char_idx]).to(self.device)
        move_logits = self.move_net(board_input, misc_input, selected_char_idx_tensor)[0]  # (17, 13)
        move_mask = torch.full_like(move_logits, float('-inf'))
        valid_moves = engine.valid_move(engine.characters[selected_char_idx])
        for y, x in valid_moves:
            move_mask[y, x] = move_logits[y, x]
        move_probs = F.softmax(move_mask.view(-1), dim=0)
        move_flat = torch.argmax(move_probs).item()

        return divmod(move_flat, 13)  # (y, x)

    def search_power(self, engine, opened_characters, next_turn, char_name):
        board_input, misc_input = self.extract_state(engine, opened_characters, next_turn)
        board_input = board_input.to(self.device)
        misc_input = misc_input.to(self.device)
        raw_logits = self.power_net(board_input, misc_input, char_name)[0]
        if char_name == 'Smith':
          masked_logits, smith_action_space = self.mask_smith_power_logits(raw_logits, engine)
          selected_idx = torch.argmax(F.softmax(masked_logits, dim=-1)).item()
          vi, ei = smith_action_space[selected_idx]

          return {
              'turn_off': engine.board.light[vi],
              'turn_on': engine.board.light[ei]
          }

        if char_name == 'Bert':
              masked_logits, bert_action_space = self.mask_bert_power_logits(raw_logits, engine)
              selected_idx = torch.argmax(F.softmax(masked_logits, dim=-1)).item()
              vi, ei = bert_action_space[selected_idx]

              return {
                  'close_hole': engine.board.hole[vi],
                  'open_hole': engine.board.hole[ei]
              }

        if char_name == 'Lestrade':
            masked_logits, lestrade_action_space = self.mask_lestrade_power_logits(raw_logits, engine)
            selected_idx = torch.argmax(F.softmax(masked_logits, dim=-1)).item()
            vi, ei = lestrade_action_space[selected_idx]

            return {
                'close_exit': engine.board.exit[vi],
                'open_exit': engine.board.exit[ei]
            }

        if char_name == 'Watson':
            selected_idx = torch.argmax(F.softmax(raw_logits, dim=-1)).item()

            return selected_idx

        return None

if __name__ == '__main__':
  from env import Engine

  engine = Engine()
  detector = Detector_agent()
  idx, name, order = detector.search_character_and_order(engine, ['Bert', 'Smith', 'Lestrade'], True)
  print(idx, name, order)
  print(detector.search_move(engine, ['Bert', 'Smith', 'Lestrade'], True, idx))
  print(detector.search_power(engine, ['Bert', 'Smith', 'Lestrade'], True, name))