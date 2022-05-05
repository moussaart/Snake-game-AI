from statistics import mean
import torch
import random
import numpy as np 
from collections import deque
from game import *
from model import Linar_QNet,QTrainer
from helper import *

MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

# creat agent class 

class AGENT :
      def __init__(self) :
            self.n_game=0
            self.epsilon=0 # randomness
            self.gamma = 0.9 # discount rate
            self.memory=deque(maxlen=MAX_MEMORY) # popleft
            self.model= Linar_QNet(11,256,3)
            self.trainer=QTrainer(self.model,lr=LR,gamma=self.gamma)
            
          
      def get_state(self,game) :
          head=game.snake[0]

          point_l=Point(head.x-20,head.y) # 20 is th block size in the class SankeGameAI
          point_r=Point(head.x+20,head.y) 
          point_u=Point(head.x,head.y-20)
          point_d=Point(head.x,head.y+20)

          dir_l=game.direction==Direction.LEFT
          dir_r=game.direction==Direction.RIGHT
          dir_u=game.direction==Direction.UP
          dir_d=game.direction==Direction.DOWN

          state=[
                #Denger straight
                (dir_r and game._is_collision(point_r)) or
                (dir_l and game._is_collision(point_l)) or
                (dir_u and game._is_collision(point_u)) or 
                (dir_d and game._is_collision(point_d)) ,
                #Denger right
                (dir_u and game._is_collision(point_r)) or
                (dir_d and game._is_collision(point_l)) or
                (dir_l and game._is_collision(point_u)) or 
                (dir_r and game._is_collision(point_d)) ,

                #Denger left
                (dir_d and game._is_collision(point_r)) or
                (dir_u and game._is_collision(point_l)) or
                (dir_l and game._is_collision(point_u)) or 
                (dir_r and game._is_collision(point_d)) ,

                #MOve direction 
                dir_l ,
                dir_r ,
                dir_u ,
                dir_d ,

                #Food location

                game.food.x < game.head.x, # food left
                game.food.x > game.head.x, # food right
                game.food.y < game.head.y, # food up
                game.food.y > game.head.y # food down

          ]
          return np.array(state,dtype=int)

      def remember(self,state,action,reward,next_state,done):
            self.memory.append((state,action,reward,next_state,done)) # popleft if MAX_MEMRY is reached

      def train_long_memory(self) :
            if len(self.memory)<BATCH_SIZE :
                  mini_sample=random.sample(self.memory,BATCH_SIZE) #list of tuples
            else :
                  mini_sample=self.memory
            
            states,actions,rewards,next_states,dones=zip(*mini_sample)
            self.trainer.train_step(states,actions,rewards,next_states,dones)
            #for states,actions,rewards,next_states,dones in mini_sample :
            #      self.trainer.train_step(states,actions,rewards,next_states,dones)


      def train_short_memory(self,state,action,reward,next_state,done) :
            self.trainer.train_step(state,action,reward,next_state,done)
      

      def get_action(self,state) :
            #random moves : tradeoff exploration / exploition
            self.epsilon=80-self.n_game
            final_move=[0,0,0]
            if random.randint(0,200)<self.epsilon :
                  move = random.randint(0,2)
                  final_move[move]=1
            else : 
                  state0=torch.tensor(state,dtype=torch.float)
                  prediction=self.model.forward(state0)
                  move=torch.argmax(prediction).item()
                  final_move[move]=1
            return final_move

def train() :
      plot_scores=[]
      plot_maen_scores=[]
      totol_scor=0
      record=0
      agent =AGENT()
      game=SnakeGameAI()
      while True :
            #get  old state 
            state_old=agent.get_state(game)

            # get move
            final_move=agent.get_action(state_old)

            #prform mobe and get new state
            reward,done,score=game.play_step(final_move)
            state_new=agent.get_state(game)

            # train short memory 
            agent.train_short_memory(state_old,final_move,reward,state_new,done)

            #remember 
            agent.remember(state_old,final_move,reward,state_new,done)
            if done :
              # train lon memory ,plot result
              game.reset()
              agent.n_game+=1
              agent.train_long_memory()
              if score > record :
                    record=score
                    agent.model.save()
              print('Game' , agent.n_game,'Score' , score , 'Record : ',record)

              #plot 
              plot_scores.append(score)
              totol_scor+=score
              mean_score=totol_scor/agent.n_game
              plot_maen_scores.append(mean_score)
              plot(scores=plot_scores,mean_scores=plot_maen_scores)


               
if __name__=='__main__' :
      train()
