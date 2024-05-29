import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # If we exceed the memory, it will remove elements from the left
        # TODO: model & training

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, game_over):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        pass


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score, record = 0, 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get the current state
        old_state = agent.get_state(game)
        # Get move
        final_move = agent.get_action(old_state)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # Remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            # Train long memory with all previous moves, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                # ToDo: agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)


    pass


if __name__ == '__main__':
    train()
