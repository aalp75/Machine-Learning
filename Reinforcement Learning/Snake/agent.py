import numpy as np
import random
import time
from game import SnakeGameAI, Direction, Point
from Qlearn import Qlearn
from NN import DQN


class Agent:
    
    def __init__(self, mode ="train", model_type ="DQN"):
        
        self.nb_games = 0
        self.epsilon = 0
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.ite = 0
        self.record = 0
        self.mode = mode
        
        self.model_type = model_type
        if self.model_type == "DQN":
            self.model = DQN( self.learning_rate)
        if self.model_type == "Qlearn":
            self.model = Qlearn()
        
        if self.mode == "play":
            self.model.load()
            self.epsilon = 0
        else:#train
            self.epsilon = 0.8
            
        self.state_memory = []
        self.action_memory = []
        self.next_state_memory = []
        self.reward_memory = []
        self.done_memory = []
        
        self.memory_size = 0
        
        self.max_memory = 10000
        
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
        action = [0,0,0]
        if (U < self.epsilon):
            action[random.randrange(0,2)] = 1
            return action
        else:
            if self.model_type =="DQN":
                action[np.argmax(self.model.predict(state))]=1
                return action
            else:#!model = Qlearn
                return self.model.predict(state)
               
    def replay(self,batch_size = 32):#Replay experience
        
        batch = np.random.choice(len(self.state_memory), batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.done_memory[batch]
        
        targets = self.model.predict(states)
        target_f = self.model.predict(next_states)
        
        for idx in range(batch_size):
            y = rewards[idx][0]
            if not dones[idx]:
                y += self.discount_factor * max(target_f[idx])       
            targets[idx][np.argmax(actions[idx])] = y
        
        self.model.model.fit(states,targets,verbose=0)
        

    def train(self, state, action, reward, next_state, done):
        if self.mode == "train":
            if self.model_type =="Qlearn":
                self.model.train(state, action, next_state,  reward, self.learning_rate, self.discount_factor)
            else:
                self.remember(state,action,reward,next_state,done)
                if (self.memory_size > 32):
                    self.replay()
        
    def remember(self,state,action,reward,next_state,done):
        
        self.memory_size +=1
        
        action = np.reshape(action,((1,3)))
        reward = np.reshape(reward,(1,1))
        done = np.reshape(done,(1,1))
        
        if self.memory_size == 1:
            self.state_memory = state
            self.action_memory = action
            self.reward_memory = reward
            self.next_state_memory = next_state
            self.done_memory = done       
            return
            
        if self.memory_size <= self.max_memory:
            self.state_memory = np.concatenate((self.state_memory,state),axis=0)
            self.action_memory = np.concatenate((self.action_memory,action),axis=0)   
            self.reward_memory = np.concatenate((self.reward_memory,reward),axis=0) 
            self.next_state_memory = np.concatenate((self.next_state_memory,next_state),axis=0) 
            self.done_memory = np.concatenate((self.done_memory,done) , axis=0)
            
        else: #memory_size > max_memory
            index = self.memory_size % self.max_memory
            
            self.state_memory[index] =state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.next_state_memory[index] = next_state
            self.done_memory[index] = done
        
    def save(self):
        if self.mode == "train":
            self.model.save()
        
        
    def load(self):
        self.model.load()
    
def play(mode,model):
    
    #Expantional moving average
    mean_score = 0
    alpha = 0.95
    
    #Create the game and the agent
    agent = Agent(mode,model)
    game = SnakeGameAI()
    current_time = time.time()

    while True:
        state = agent.get_state(game)
        if agent.model_type == "DQN":#Reshape in this case to be in good format for the tf NN
            state = np.reshape(state,(1,11))
        
        action = agent.get_move(state)
        
        reward, done, score = game.play_step(action)
        
        next_state = agent.get_state(game)
        if agent.model_type == "DQN":
            next_state = np.reshape(next_state,(1,11))
            
        agent.train(state,action,reward,next_state,done)
        
        state = next_state
        
        
        if done:
            agent.ite +=1
            agent.epsilon = agent.epsilon * 0.9#Diminue the epsilon exploration parameter
            
            if (game.score > agent.record):#Best result
                agent.record = game.score
                agent.save()
            
            mean_score = round(alpha * mean_score + (1 - alpha)*score,2)
            timer = round(time.time() - current_time,2)
            
            print("Game", agent.ite ,"Score", game.score , "Record", agent.record,"EMA", mean_score,\
            "Time", timer , 'seconds')
                
            game.reset()#Launch new game

MODE = "train"
#MODE= "play"  

#MODEL = "DQN"      
MODEL = "Qlearn"
         
if __name__ == '__main__':
    play(MODE,MODEL)
        
        
