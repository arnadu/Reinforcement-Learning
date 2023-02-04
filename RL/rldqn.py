import math
import time
import random
from collections import deque
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  #activation function https://pytorch.org/docs/stable/nn.functional.html
from torch.profiler import profile, record_function, ProfilerActivity

import gym

#--------------------------------
class SimpleMLP(nn.Module):

    def __init__(self, num_observations, num_actions, num_neurons):
        super(SimpleMLP, self).__init__()
        
        self.layer1 = nn.Linear(num_observations, num_neurons)
        self.layer2 = nn.Linear(num_neurons, num_neurons)
        self.layer3 = nn.Linear(num_neurons, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


#--------------------------------
#--------------------------------

Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'done', 'next_obs'))

class ReplayMemory():
    
    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)
    
    def append(self, transition: Transition):
        self.memory.append(transition)
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))        #see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html for the *zip* trick
    
    def __len__(self):
        return len(self.memory)

TrainingRecord = namedtuple('Training', 
                            ('t', 'episode', 'episode_t', 'epsilon', 'score', 'loss_mean', 'loss_std'))

@dataclass
class RLDQNParams:
    num_neurons: int = 32              #number of neurons in the simple MLP used to compute Q values
    max_episode_length: int = 600      #maximum length of an episode
    max_time_steps: int = 1000*600     #number of time steps used for training the model
    train_period: int = 1              #train the policy network every x timesteps
    learning_rate: float = 0.0001      #learning rate for the AdamW optimizer
    memory_size: int = 50_000          #size of replay memory, older samples are discarded
    memory_batch: int = 64             #size of batch sampled from replay memory for each training step
    gamma: float = 0.9                 #discount factor
    epsilon: float = 0.5               #initial epsilon value for the epsilon-greedy policy
    epsilon_min: float = 0.05          #minimum epsilon value for the epsilon-greedy strategy
    epsilon_half_life: int = 200       #decrease epsilon at every time step with the given half life
    target_update_rate: float = 0.05   #rate at which the target model is updated from the policy model
    log_recent_episodes: int = 100     #print a message every x episodes during training loop
    
class RLDQN():
    def __init__(self, env: gym.Env, params: RLDQNParams, device):
        
        #get input parameters
        self.params = params
        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.device = device

        self.model = SimpleMLP(num_observations = self.num_observations, num_actions=self.num_actions, num_neurons=self.params.num_neurons).to(self.device)
        self.target_model = SimpleMLP(num_observations = self.num_observations, num_actions=self.num_actions, num_neurons=self.params.num_neurons).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def play(self, env):
        obs = env.reset()
        done=False
        while not done:
            env.render()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) 
            action = self.select_action(env, obs, epsilon=0.)
            obs, reward, done, info = env.step(action.item())        
        env.close()
        
    def evaluate(self, env, num_episodes=100, episode_length=600):
        scores=[]
        for e in range(num_episodes):
            obs = env.reset()
            score=0
            done=False
            episode_t=0
            while not done:
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
                action = self.select_action(env, obs, epsilon=0.)
                obs, reward, done, info = env.step(action.item())
                score += reward
                episode_t +=1
                if episode_t>episode_length: #force episode termination
                    done=True
            scores.append(score)
        mean_reward = np.mean(scores)
        std_reward = np.std(scores)
        return mean_reward, std_reward, scores

    def train(self, env, params):
        
        self.params = params
        print("start training with params:")
        print(params)
        
        finished_training = False
        epsilon = self.params.epsilon
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params.learning_rate, amsgrad=True) #https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        criterion = nn.SmoothL1Loss() #https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

        self.memory = ReplayMemory(self.params.memory_size)
        self.training_record = [] #array of TrainingRecord namedtuples to record progress during the training loop; one entry per episode
        
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        
        t = 0 #to measure time since beginning of training
        episode = 0 #to measure number of episodes
        episode_t = 0 #to measure length of current episode
        score = 0. #to track score of current episode
        loss_record = [] #track loss during episode

        #training loop
        #environment is reset within the loop whenever it reaches a terminal step or when it reaches the max lenghth of an episode
        #tqdm_bar = tqdm(range(self.params.max_time_steps))
        while not finished_training:
           
            #generate the next transition and add it to replay memory

            action = self.select_action(env, obs, epsilon)
            next_obs, reward, done, info = env.step(action.item())

            score += reward #episode's score
            #obs is already a tensor on device
            #action is already a tensor on device, see select_action()
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0) #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            not_done = torch.tensor([0. if done else 1.], dtype=torch.float32, device=self.device)  #flip done to more easily calculate td_targets during training (see below)
            self.memory.append(Transition(obs, action, reward, not_done, next_obs))

            #train the model

            if len(self.memory) > self.params.memory_batch and t % self.params.train_period == 0:

                batch = self.memory.sample(self.params.memory_batch)

                #calculate the current model's estimate of the Q value corresponding to the observation and action taken at the time
                #Q = model[obs][action]
                q = self.model(torch.cat(batch.obs)).gather(1, torch.cat(batch.action)).squeeze(1) 

                #calculate the temporal difference's target value, which is the reward + the discounted Q value of the following state
                #use the 'target' model for stability, the target model evolves more slowly than the policy model
                #non_terminal is 0 if the transition was terminal, 1 otherwise (this is the flip of the 'done' value returned by the env
                #TD Target = reward + non_terminal * gamma * max_action[model(next_obs)]
                with torch.no_grad():
                    td_targets = self.target_model(torch.cat(batch.next_obs)) #calculate the Q values on the resulting state of the transition
                    td_targets = td_targets.max(1).values #get the max Q value across possible actions
                    td_targets *= torch.cat(batch.done) #keep target Q value estimate only for non-terminal transition .done is 0 is we reached the terminal state at this transition, 1 otherwise (see how the done flag is inverted when recording it in replay memory
                    td_targets *= self.params.gamma #discount
                    td_targets += torch.cat(batch.reward) #add reward for both terminal and non-terminal transitions

                #calculate the temporal difference loss
                #the criterion function returns the average loss over the transitions sampled in this batch
                #the loss function is the square of the error is the error is less than one, or the abs value of the error otherwise
                #this is more robust to outliers than pure squared error
                #td_loss = mean_batch[(td_targets-q)^2]
                td_loss = criterion(td_targets, q)
                loss_record.append(td_loss.item())

                #update the model's parameters to minimize td loss by stochastic gradient descent
                optimizer.zero_grad()
                td_loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(),100) #https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
                optimizer.step()

                #update the parameters theta' of the target model from the policy model
                #theta' <- update_rate*theta + (1-update_rate)*theta'
                with torch.no_grad():
                    target_params = self.target_model.state_dict() #https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
                    policy_params = self.model.state_dict()
                    for key in policy_params:
                        target_params[key] = policy_params[key]*self.params.target_update_rate + target_params[key]*(1. - self.params.target_update_rate)
                    self.target_model.load_state_dict(target_params)


            episode_t += 1
            if done or episode_t>self.params.max_episode_length: 

                #end of episode, prepare next one

                if epsilon > self.params.epsilon_min:
                    epsilon -= epsilon * math.log(2)/self.params.epsilon_half_life
                
                #log temporary results
                loss_mean = np.mean(loss_record)
                loss_std = np.std(loss_record)
                loss_record=[]
                r = TrainingRecord(t=t, episode=episode, episode_t=episode_t, epsilon=epsilon, score=score, loss_mean=loss_mean, loss_std=loss_std)
                self.training_record.append(r)
                
                if episode % self.params.log_recent_episodes==0:
                    print(r)

                #start a new episode
                #tqdm_bar.set_postfix({'episode': episode})
                episode += 1
                episode_t = 0
                score = 0.
                next_obs = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0) 

            #prepare next iteration
            
            obs = next_obs

            t += 1
            if t>self.params.max_time_steps:
                finished_training=True  #will exit the main training loop

        

    def select_action(self, env, obs, epsilon):
        if random.random()<epsilon:
            action = env.action_space.sample()
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.model(obs)
                max_q = q_values.max(1).indices
                return max_q.view(1,1)

            
      





