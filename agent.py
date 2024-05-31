import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

import snake_game
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # If we exceed the memory, it will remove elements from the left
        self.model = Linear_QNet(11, 256, 3)  # Size of the state, hidden neurons, output action
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    @staticmethod
    def get_state(game):
        """
        There are 11 possible values that encode where a collision would occur, the snake direction and the position of
        the food with respect to the snake.
        :param game: an instance of a SnakeGameAI
        :return: binary encoding of the current game state
        """
        def danger_straight(right_direction: bool, left_direction: bool, up_direction: bool, down_direction: bool,
                            point_r: Point, point_l: Point, point_u: bool, point_d: Point,
                            game_object: SnakeGameAI = game) -> bool:
            """
            Determines whether a danger is placed on the current direction of the snake.

            :param right_direction: boolean that encodes if the current snake directions is `right`
            :param left_direction: boolean that encodes if the current snake directions is `left`
            :param up_direction: boolean that encodes if the current snake directions is `up`
            :param down_direction: boolean that encodes if the current snake directions is `down`
            :param point_r: coordinates of the right boundary of the snake head
            :param point_l: coordinates of the left boundary of the snake head
            :param point_u: coordinates of the upper boundary of the snake head
            :param point_d: coordinates of the lower boundary of the snake head
            :param game_object: an instance of a SnakeGameAI

            :return: boolean value encoding if a danger is placed on the current direction
            """
            danger_is_straight = (right_direction and game_object.is_collision(point_r)) or \
                                 (left_direction and game_object.is_collision(point_l)) or \
                                 (up_direction and game_object.is_collision(point_u)) or \
                                 (down_direction and game_object.is_collision(point_d))

            return danger_is_straight

        def danger_right(right_direction: bool, left_direction: bool, up_direction: bool, down_direction: bool,
                         point_r: Point, point_l: Point, point_u: bool, point_d: Point,
                         game_object: SnakeGameAI = game) -> bool:
            """
            Determines whether a danger is right placed with respect to the current direction of the snake

            :return: boolean value encoding if a danger is placed on the right of the current direction
            """
            danger_is_right = (right_direction and game_object.is_collision(point_d)) or \
                              (left_direction and game_object.is_collision(point_u)) or \
                              (up_direction and game_object.is_collision(point_r)) or \
                              (down_direction and game_object.is_collision(point_l))

            return danger_is_right

        def danger_left(right_direction: bool, left_direction: bool, up_direction: bool, down_direction: bool,
                         point_r: Point, point_l: Point, point_u: bool, point_d: Point,
                         game_object: SnakeGameAI = game) -> bool:
            """
            Determines whether a danger is left placed with respect to the current direction of the snake

            :return: boolean value encoding if a danger is placed on the left of the current direction
            """
            danger_is_right = (right_direction and game_object.is_collision(point_u)) or \
                              (left_direction and game_object.is_collision(point_d)) or \
                              (up_direction and game_object.is_collision(point_l)) or \
                              (down_direction and game_object.is_collision(point_r))

            return danger_is_right

        head = game.snake[0]

        # Coordinates of the head boundaries to check where is the danger with repect to it
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)

        # Encode the current snake direction
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [danger_straight(dir_right, dir_left, dir_up, dir_down, point_right, point_left, point_up, point_down),
                 danger_right(dir_right, dir_left, dir_up, dir_down, point_right, point_left, point_up, point_down),
                 danger_left(dir_right, dir_left, dir_up, dir_down, point_right, point_left, point_up, point_down),
                 # Move direction encoding:
                 dir_left,
                 dir_right,
                 dir_up,
                 dir_down,
                 # Reward (food) location:
                 game.food.x < game.head.x,  # Reward to the left
                 game.food.x > game.head.x,  # Reward to the right
                 game.food.y < game.head.y,  # Reward above
                 game.food.y > game.head.y,  # Reward below
                 ]

        return np.array(state, dtype=int)  # Convert boolean values to binary

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Random Moves: Tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games  # The more games we have, the smaller epsilon gets
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # Exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # Exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # This will execute the forward function within the model
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


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
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
