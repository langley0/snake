import pygame
import numpy as np
import random
import time

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
ACTION_DOWN = 3

class SnakeGame(object):
    def __init__(self, width = 20, height = 20):
        self.width = width
        self.height = height
        self.player_board = np.zeros(self.width * self.height)
        self.object_board = np.zeros(self.width * self.height)

        # player locate
        start_pos = int((height + 1 ) * width / 2)
        self.player_chain = [start_pos]
        self.player_size = 1
        self.player_board[start_pos] = 1

        self.player_direction =  ACTION_LEFT

        self.locate_new_object()
        
    def locate_new_object(self):
        board = np.zeros(self.width * self.height)
        # locate target object in current map
        candidates = []
        for i in range(self.width*self.height):
            if self.player_board[i] == 0:
                candidates.append(i)
        
        target = random.choice(candidates)
        board[target] = 1
        self.object_board = board

    def check_hit(self):
        for pos in self.player_chain:
            if self.object_board[pos] == 1:
                return True

        return False
        
    def move(self, nx, ny):
        new_head = ny * self.width + nx
        new_chain = [new_head] + self.player_chain[:]
        self.player_chain = new_chain[:self.player_size]
        board = np.zeros(self.width * self.height)
        for pos in self.player_chain:
             board[pos] = 1
        self.player_board = board

    def get_player_input(self):
        for event in pygame.event.get():
            if (pygame.QUIT == event.type):
                exit()
            
            if (pygame.KEYDOWN == event.type):   
                if (pygame.K_ESCAPE == event.key):   
                    exit()
                elif (pygame.K_UP == event.key):
                    self.player_direction = ACTION_UP
                elif (pygame.K_DOWN == event.key):
                    self.player_direction = ACTION_DOWN
                elif (pygame.K_LEFT == event.key):
                    self.player_direction = ACTION_LEFT
                elif (pygame.K_RIGHT == event.key):
                    self.player_direction = ACTION_RIGHT


    def update(self):
        # check crash
        head = self.player_chain[0]
        x = head % self.width
        y = int(head / self.width)
        if self.player_direction == ACTION_LEFT:
            x -= 1
        elif self.player_direction == ACTION_RIGHT:
            x += 1
        elif self.player_direction == ACTION_UP:
            y -= 1
        elif self.player_direction == ACTION_DOWN:
            y += 1
        
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return -1

        # check self hit
        if self.player_board[y * self.width + x] == 1:
            # self hit
            return -1

        # move
        self.move(x, y)

        # check player getting object 
        if self.check_hit():
            self.locate_new_object()
            self.player_size += 1
            return 1
        
        # nothing happened
        return 0

    def draw(self, screen=None):
        if screen is None:
            return
        
        screen.fill((10,10,10))

        size = 300 / max(self.width, self.height) 

        for y in range(self.height):
            for x in range(self.width):
                offset = y * self.width + x
                if self.player_board[offset] == 1:
                    pygame.draw.rect(screen, (100,220,100), (x*size, y * size, size, size))
                elif self.object_board[offset] == 1:
                    pygame.draw.rect(screen, (220,100,100), (x*size, y * size, size, size))

        pygame.display.flip()
                

def init():
    # screen init
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption('Omok')

    # create gackground
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250,250,250))

    return screen

def main(game, screen=None):

    _t = time.time()
    score = 0 
    while True:
        game.draw(screen)
        # check player input
        game.get_player_input()
        # update
        if (time.time() - _t > 0.1):
            reward = game.update()
            score += reward 
            if reward < 0:
                return score
            _t = time.time()
        
if __name__ == '__main__':
    screen = init()
    game = SnakeGame()
    score = main(game, screen)
    print("score is %d" % score)