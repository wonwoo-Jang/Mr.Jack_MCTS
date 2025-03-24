import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from value_model import JackValueModel
from utils import generate_all_possible_next_states

class JackAgent:
    def __init__(self):
        self.value_model = JackValueModel()
        self.name = 'jack'

    def select_best_next_state(self, engine):
        next_states = generate_all_possible_next_states(engine, mode=self.name)
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