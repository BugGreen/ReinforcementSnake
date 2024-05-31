import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()  # Initialize all modules correctly
font = pygame.font.Font('arial.ttf', 25)
InitialPoint = namedtuple('InitialPoint', 'x, y')  # InitialPoint has members `x` and `y`
Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
BLOCK_SIZE = 20
SPEED = 10


class Direction(Enum):  # Set symbolic names bounded to unique values
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.direction = Direction.RIGHT  # Snake starts moving to the right
        self.head = InitialPoint(self.w / 2, self.h / 2)
        self.snake = [self.head, InitialPoint(self.head.x - BLOCK_SIZE, self.head.y),
                      InitialPoint(self.head.x - (2 * BLOCK_SIZE), self.head.y)]  # Initial snake body
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        """
        Generate food in random places of the display
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # Do not place food inside the snake
        if self.food in self.snake:
            self._place_food()  # Place food in another coordinates

    def _update_ui(self):
        """
        Update the display when an event occur
        :return: None
        """
        self.display.fill(BLACK)
        # Draw the snake
        for point in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x + 4, point.y + 4, 12, 12))
        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Draw the score
        text = font.render('Score: ' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Update the whole display

    def _move(self, direction: dict):
        """
        Updates the coordinates of the snake
        :param direction: command given by the player
        :return: None
        """
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _is_collision(self) -> bool:
        """
        Define collision events
        :return: If collision occurs
        """
        x = self.head.x
        y = self.head.y
        # Hits boundary
        if x > self.w - BLOCK_SIZE or x < 0 or y > self.h - BLOCK_SIZE or y < 0:
            return True
        # Hits itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _collect_user_input(self) -> str:
        """
        Map player input into a direction command for the snake
        :return: direction previous to the user input
        """
        prev_direction = self.direction
        opposite_direction = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.UP,
            Direction.DOWN: Direction.UP
        }
        for event in pygame.event.get():  # Get all user events
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        self.direction = prev_direction if opposite_direction.get(self.direction) == prev_direction else self.direction
        return prev_direction

    def play_step(self):
        # Collect user input
        direction_bef_input = self._collect_user_input()

        # Move
        self._move(self.direction)
        self.snake.insert(0, self.head)  # Update the head at the beginning of the snake list

        # Check game over status
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        # Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)
        # Return game over and score
        game_over = False
        return game_over, self.score


if __name__ == '__main__':
    game = SnakeGame()

    # Game loop
    while True:
        game_over, score = game.play_step()

        # Break if game over
        if game_over:
            break
    pygame.quit()

