import numpy as np
import random
import time
from game import SnakeGameAI, Direction, Point
from Qlearn import Qlearn


class Agent:
    
    def __init__(self, mode ="train"):
        self.nb_games = 0
        self.model = Qlearn()
        self.epsilon = 0
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.ite = 0
        self.record = 0
        self.mode = mode
        
        if self.mode == "play":
            self.model.load()
            self.epsilon = 0
        else:#train
            self.epsilon = 0.9
        
    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - game.block_size, head.y)
        point_r=Point(head.x + game.block_size, head.y)
        point_u=Point(head.x, head.y - game.block_size)
        point_d=Point(head.x, head.y + game.block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, # food is in left
            game.food.x > game.head.x, # food is in right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int)
        
        
    def get_move(self,state):
        U = random.uniform(0,1)
        if (U < self.epsilon):
            action = [0,0,0]
            action[random.randrange(0,2)] = 1
            return action
        else:
            return self.model.predict(state)
    
    def train(self,old_state, action, new_state,reward):
        if self.mode == "train":
            self.model.train(old_state, action, new_state,  reward, self.learning_rate, self.discount_factor)
        
    def save(self):
        if self.mode == "train":
            self.model.save()
        
    def load(self):
        self.model.load()
    
def play(mode):
    agent = Agent(mode)
    game = SnakeGameAI()
    current_time = time.time()

    while True:
        state = agent.get_state(game)
        move = agent.get_move(state)
        reward, done, score = game.play_step(move)
        
        next_state = agent.get_state(game)
        
        agent.train(state,move,next_state,reward)
        
        state = next_state
        
        if done:
            agent.ite +=1
            agent.epsilon = agent.epsilon * 0.99
            
            if (game.score > agent.record):
                agent.record = game.score
                agent.save()
                
            timer = round(time.time() - current_time,2)
            print("Game", agent.ite ,"Score", game.score , "Record", agent.record,\
            "Time", timer , 'seconds')
            game.reset()

mode = "train"
#mode= "play"        
         
if __name__ == '__main__':
    play(mode)
        
        
