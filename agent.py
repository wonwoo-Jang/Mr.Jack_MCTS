import torch
from value_model import DetectorValueModel, JackValueModel
from utils import generate_all_possible_next_states
import random

class DetectorAgent:
    def __init__(self, model_path=None):
        self.value_model = DetectorValueModel()
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            self.value_model.load_state_dict(state_dict)
        self.name = 'detector'

    def select_best_next_state(self, engine, epsilon=0):
        next_states = generate_all_possible_next_states(engine, mode=self.name)

        # ε 확률로 랜덤하게 선택
        if random.random() < epsilon:
            return random.choice(next_states)

        # 그 외에는 greedy하게 선택
        best_state = None
        best_value = float('-inf')

        for next_engine in next_states:
            board_input, misc_input = next_engine.extract_state(mode=self.name)
            with torch.no_grad():
                value = self.value_model(board_input, misc_input).item()

            if value > best_value:
                best_value = value
                best_state = next_engine

        return best_state

class JackAgent:
    def __init__(self, model_path=None):
        self.value_model = JackValueModel()
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            self.value_model.load_state_dict(state_dict)
        self.name = 'jack'

    def select_best_next_state(self, engine, epsilon=0):
        next_states = generate_all_possible_next_states(engine, mode=self.name)

        # ε 확률로 랜덤하게 선택
        if random.random() < epsilon:
            return random.choice(next_states)

        # 그 외에는 greedy하게 선택
        best_state = None
        best_value = float('-inf')

        for next_engine in next_states:
            board_input, misc_input = next_engine.extract_state(mode=self.name)
            with torch.no_grad():
                value = self.value_model(board_input, misc_input).item()

            if value > best_value:
                best_value = value
                best_state = next_engine

        return best_state

if __name__ == '__main__':
    from env import Engine

    engine = Engine()
    jack = JackAgent()
    print(engine.extract_state(mode='jack'))
    print(jack.select_best_next_state(engine).extract_state(mode='jack'))