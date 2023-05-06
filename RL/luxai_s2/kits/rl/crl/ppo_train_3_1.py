# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:04:46 2023

@author: remyh

BASED on 
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

"""


###Add one feature to observation space, with the water cargo of the mother factory
###this is defined in SimpleUnitObservationWrapper_1
###hopefully this will give a signal to the unit about when to stop digging and return to the factory to deliver water

import argparse
import os
import random
import time
from distutils.util import strtobool
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice


from wrappers import SimpleUnitDiscreteController_3
from my_sync_vector_env import MySyncVectorEnv

####################################################
####################################################
###from luxai_s2/wrappers/sb3.py
from typing import Callable, Dict
import numpy.typing as npt
import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import (
    Controller,
)

#same as example sb3 wrapper
#changed call to actions_to_lux_actions() from SimpleUnitDiscreteController_3 to handle multiple units
#added call to factory_actions() from SimpleUnitDiscreteController_3 to create light units
class SinglePhaseWrapper(gym.Wrapper):  
    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[
            [str, ObservationStateDict], Dict[str, BidActionType]
        ] = None,
        factory_placement_policy: Callable[
            [str, ObservationStateDict], Dict[str, FactoryPlacementActionType]
        ] = None,
        controller: Controller = None,
    ) -> None:
        """
        A environment wrapper for Stable Baselines 3. It reduces the LuxAI_S2 env
        into a single phase game and places the first two phases (bidding and factory placement) into the env.reset function so that
        interacting agents directly start generating actions to play the third phase of the game.
        It also accepts a Controller that translates action's in one action space to a Lux S2 compatible action
        Parameters
        ----------
        bid_policy: Function
            A function accepting player: str and obs: ObservationStateDict as input that returns a bid action
            such as dict(bid=10, faction="AlphaStrike"). By default will bid 0
        factory_placement_policy: Function
            A function accepting player: str and obs: ObservationStateDict as input that returns a factory placement action
            such as dict(spawn=np.array([2, 4]), metal=150, water=150). By default will spawn in a random valid location with metal=150, water=150
        controller : Controller
            A controller that parameterizes the action space into something more usable and converts parameterized actions to lux actions.
            See luxai_s2/wrappers/controllers.py for available controllers and how to make your own
        """
        gym.Wrapper.__init__(self, env)
        self.env = env
        
        assert controller is not None
        
        # set our controller and replace the action space
        self.controller = controller
        self.action_space = controller.action_space

        # The simplified wrapper removes the first two phases of the game by using predefined policies (trained or heuristic)
        # to handle those two phases during each reset
        if factory_placement_policy is None:
            def factory_placement_policy(player, obs: ObservationStateDict):
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
                )
                spawn_loc = potential_spawns[
                    np.random.randint(0, len(potential_spawns))
                ]
                return dict(spawn=spawn_loc, metal=150, water=150)

        self.factory_placement_policy = factory_placement_policy
        if bid_policy is None:
            def bid_policy(player, obs: ObservationStateDict):
                faction = "AlphaStrike"
                if player == "player_1":
                    faction = "MotherMars"
                return dict(bid=0, faction=faction)

        self.bid_policy = bid_policy

        self.prev_obs = None

    def step(self, action: Dict[str, npt.NDArray]):
        
        # here, for each agent in the game we translate their action into a Lux S2 action
        lux_action = dict()
        for agent in self.env.agents:
            if agent in action:
                
                lux_action[agent] = self.controller.actions_to_lux_actions(agent=agent, obs=self.prev_obs, actions=action[agent])
                factory_actions = self.controller.factory_actions(agent, obs=self.prev_obs)
                lux_action[agent].update(factory_actions)
            else:
                lux_action[agent] = dict()
        
        # lux_action is now a dict mapping agent name to an action
        obs, reward, done, info = self.env.step(lux_action)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        # we upgrade the reset function here
        
        # we call the original reset function first
        obs = self.env.reset(**kwargs)
        
        # then use the bid policy to go through the bidding phase
        action = dict()
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)
        
        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent] = self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, _, _, _ = self.env.step(action)   
        
        self.prev_obs = obs
        
        return obs


####################################################
####################################################
#prepare a list of observation vectors, one for each unit, that can be used by the NN to find the best action for each unit

#length of the observation vector for a single robot;
#13 is the size of the original example code
#-3 for cargo (all cargo types summed up into a single number instead)
#-1 for unit type
#-1 for team id
#-2 for closest ice tile
# +9 for ice neighborhood i the 3x3 area centered on the unit
# +4 for ice in the 4 quadrants (north, south, east, west)

SHAPE = 13-3-1-1-2+9+4

def obs_for_units(obs, player, units_factories, env_cfg):

    shared_obs = obs['player_0']
    ice_map = shared_obs["board"]["ice"]
    ice_tile_locations = np.argwhere(ice_map == 1)
    padded_ice_map = np.pad(ice_map,(1,1)) #pad the matrix to facilitate the lookup of 3x3 neighborhoods
    total_ice = np.count_nonzero(ice_tile_locations) #count of ice on the board, for normalizing
    
    factories = shared_obs['factories'][player]
    factories_locations = np.array([ np.array(factories[factory_id]['pos']) for factory_id in factories.keys()])

    units_ids = []    #return the list of units, to make sure we preserve order    
    observations = [] #return an array of observations, one for each unit
    
    units = shared_obs["units"][player]
    for k in units.keys():
        unit = units[k]

        obs_vec = np.zeros(SHAPE) #size of observation space for NN model
            
        # store cargo+power values scaled to [0, 1]
        cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
        battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
        cargo_vec = np.array(
            [
                unit["power"] / battery_cap,
                (unit["cargo"]["ice"]+
                 unit["cargo"]["ore"]+
                 unit["cargo"]["water"]+
                 unit["cargo"]["metal"]) / cargo_space 
#                    unit["cargo"]["ore"] / cargo_space,
#                    unit["cargo"]["water"] / cargo_space,
#                    unit["cargo"]["metal"] / cargo_space,
            ]
        )

#            unit_type = (
#                0 if unit["unit_type"] == "LIGHT" else 1
#            )  # note that build actions use 0 to encode Light

        # normalize the unit position
        pos = np.array(unit["pos"]) / env_cfg.map_size
        
        unit_vec = np.concatenate([pos, cargo_vec, ], axis=-1)
#            unit_vec = np.concatenate([pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1)

        # we add some engineered features down here

        # compute closest ice tile
#            ice_tile_distances = np.mean((ice_tile_locations - np.array(unit["pos"])) ** 2, 1)
#            closest_ice_tile = (ice_tile_locations[np.argmin(ice_tile_distances)] / self.env_cfg.map_size)

        #get the ice inventory in the immediate neighborhood, and the 4 quadrants
        x, y = unit['pos'][0]-1, unit['pos'][1]-1 #top left corner of the 3x3 neighborhood centered on the unit
        
        neighborhood_ice = padded_ice_map[np.ix_([x+1,x+2,x+3],[y+1,y+2,y+3])].flatten()  
        
        north_ice = ice_map[:,:y].sum()  #total ice north of the unit's immediate neighborhood
        south_ice = ice_map[:,y+3:].sum()
        west_ice = ice_map[:x,:].sum()  
        east_ice = ice_map[x+3:,:].sum() 
        quadrants_ice = np.array([north_ice, south_ice, east_ice, west_ice])/total_ice #normalize with a reasonable scale


        '''
        '''
        #tell the unit to focus on its mother factory
        if (k not in units_factories) \
            or (units_factories[k] not in factories):  
            #the unit has just been created and is not yet associated with a factory
            #or the mother factory has died and we need to focus on a different one
            
            # identify the closest factory
            factories_distances = np.mean(
                (factories_locations - np.array(unit["pos"])) ** 2, 1
            )
            i = np.argmin(factories_distances)
            factory_id = list(factories)[i]  #rely on the fact that dictionaries are order-preserving
            units_factories[k]=factory_id
        
        #we want the unit to focus on this specific factory and not wander off
        factory_id = units_factories[k]
        factory = factories[factory_id] 
        factory_vec = factory['pos']/env_cfg.map_size

        #get the water stock of the mother factory as an additional observation for this unit, normalized to a reasonable scale
        #this is the same logic as as SimpleUnitObservationWrapper_1
        #factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
        
        #put the entire observation together
#            obs_vec = np.concatenate([unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1)
        obs_vec = np.concatenate([unit_vec, factory_vec - pos, neighborhood_ice, quadrants_ice], axis=-1)
        
        #return two arrays ordered in the same way, with units keys and their corresponding observations
        units_ids.append(unit['unit_id'])  
        
        observations.append(obs_vec)

    return observations 


####################################################
####################################################
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)

        self.observation_space = spaces.Box(-999, 999, shape=(SHAPE,)) 
        
        self.prev_step_metrics = None

        self.observations = []  #array of NN-compatible observation vectors, player is axis zero, unit is axis one, obs vector for each unit is axis 3
        
        self.units_factories = {'player_0':{}, 'player_1':{}}  #to remember the relation between units and their mother factories

    def observation(self):
        obs_vec = self.observations
        return obs_vec

    def step(self, action):  #{agent: action_0, opp_agent: action_1} where action_0 and action_ are arrays (one action for every unit of each player)
        agent = "player_0"
        opp_agent = "player_1"

        #TODO: figure out whether it is OK to remove this, or will this make the training unstable?
        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 1000

        #apply the actions; this will call SinglePhaseWrapper, where the NN actions are translated to Lux actions by the controller      
        #action = {agent: [actions], opp_agent: [actions]}       action should be in this format, with one action per unit for each player
        obs, _, done, info = self.env.step(action)
        #obs = obs[agent]
        done = done[agent]

        #note: we need to guarantee that the observation returned by this wrapper will always have obs[0][0] and obs[1][]: this is needed for training
        if not done:
            self.observations = [
                obs_for_units(obs, 'player_0', self.units_factories['player_0'], self.env.env_cfg),
                obs_for_units(obs, 'player_1', self.units_factories['player_1'], self.env.env_cfg)
            ]
            if len(self.observations[0])==0:
                self.observations[0] = np.zeros(SHAPE) #return a dummy vector to avoid a crash, this could happen for example if a player has lost all its units because of collisions.
                done = True #TODO- game over, as we are not creating new units for now, 
                #TODO - but this should be removed when a player is able to create new units
            if len(self.observations[1])==0:
                self.observations[1] = np.zeros(SHAPE) #return a dummy vector to avoid a crash, this could happen for example if a player has lost all its units because of collisions.
                done = True #TODO- game over, as we are not creating new units for now NOTE: does terminating here introduces a bias in the training... 
        else:
            self.observations = [np.zeros(SHAPE), np.zeros(SHAPE)]

        
        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"])
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["lichen_produced"] = stats["generation"]["lichen"]

        # we save these two to see often the agent updates robot action queues and how often enough
        # power to do so and succeed (less frequent updates = more power is saved)
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        reward = 0
        if self.prev_step_metrics is not None:
            
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (metrics["water_produced"] - self.prev_step_metrics["water_produced"])
            lichen_produced_this_step = (metrics["lichen_produced"] - self.prev_step_metrics["lichen_produced"])

            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step #+ 100 * lichen_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):

        self.units_factories = {'player_0':{}, 'player_1':{}}  #to remember the relation between units and their mother factories

        obs = self.env.reset(**kwargs)    #this is calling SinglePhaseWrapper() and will go through the factory placement process
        
        self.prev_step_metrics = None

        #run the first step after factory placement, to create the first units before training can properly start
        
        action = {'player_0':[], 'player_1':[]}       
        obs, _, _, _ = self.step(action)
        return obs


####################################################
####################################################
def make_env(env_id, seed, idx, capture_video, run_name, max_episode_steps=1000):
    def thunk():
        
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
        
        env = SinglePhaseWrapper(env, 
                                 factory_placement_policy=place_near_random_ice,
                                 controller=SimpleUnitDiscreteController_3(env.env_cfg))  #makes single phase; run the bid and factory placement in reset()
        
        
        env = CustomEnvWrapper(env)  #makes this a single player game
        
        env = TimeLimit(env, max_episode_steps)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        return env

    return thunk


####################################################
####################################################
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentNN(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


####################################################
####################################################
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    
    parser.add_argument("--env-id", type=str, default="LuxAI_S2-v0",
        help="the id of the environment")
    
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
        help="total timesteps of the experiments")
    
    parser.add_argument("--max-episode-steps", type=int, default=1000,  #run a full episode to get lichen towards the end
        help="maximum steps per episode")
    
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    
    parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")   #=> across 4 parallel env, this will be 4000 roll out steps total
    
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    
    parser.add_argument("--num-minibatches", type=int, default=5,   #each mini batch will be 8 steps (4000 rollout / 5)
        help="the number of mini-batches")
    
    parser.add_argument("--update-epochs", type=int, default=10,   #10 epochs of training for every roll out batch
        help="the K epochs to update the policy")
    
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    
    parser.add_argument("--target-kl", type=float, default=0.05,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def batch(obs):
    
    #obs[parallel env, player=0|1, unit, feature]
    shapes = [ [len(e[0]), len(e[1])] for e in obs] #count the number of units for each player in each env
    obs_0 = torch.cat([torch.Tensor(e[0]) for e in obs]).squeeze().to(device)   #gather player 1's observations into a single array = [[parallel env, unit, feature]]
    obs_1 = torch.cat([torch.Tensor(e[1]) for e in obs]).squeeze().to(device)   #gather player 1's observations into a single array = [[parallel env, unit, feature]]
    return obs_0, obs_1, shapes
    
def unbatch(x, player, shapes, first=False):
    y = []
    i=0
    for s in shapes:
        p = x[i:i+s[player]]
        if first:
            y.append(p[0])
        else:
            y.append(p)
        i=i+s[player]
    return y

####################################################
####################################################
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv(
    envs = MySyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.max_episode_steps) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = AgentNN(envs).to(device)    #used for training
    #agent_1 = AgentNN(envs).to(device)  #used to simulate the opponent, player_1
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            obs[step] = torch.stack([torch.Tensor(o[0][0]) for o in next_obs]).to(device)  #use only the first unit of player_0 for training, collected across all parallel envs
            dones[step] = next_done

            # ALGO LOGIC: action logic
          
            #next_obs = [parallel env, player, unit, obs feature]
            
            obs_0, obs_1, shapes = batch(next_obs)
            with torch.no_grad():
                #TODO: update agent_1's weights from agent 
                actions_1, _, _, _ = agent.get_action_and_value(obs_1)  #we get the action for each unit of player 1 across all parallel envs into a single list (ravel)
                actions_0, logprob, _, value = agent.get_action_and_value(obs_0) #we get the action for each unit of player 0

                values[step] = torch.stack(unbatch(value, player=0, shapes=shapes, first=True)).flatten()
                actions[step] = torch.stack(unbatch(actions_0, player=0, shapes=shapes, first=True))
                logprobs[step] = torch.stack(unbatch(logprob, player=0, shapes=shapes, first=True))
            
                
            '''
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            '''
            
            #rebuild action across parallel envs (obs were flattned to call the NN once across all parallel envs)            
            actions_0 = unbatch(actions_0, player=0, shapes=shapes)
            actions_1 = unbatch(actions_1, player=1, shapes=shapes)
            action = [{'player_0':a0, 'player_1':a1} for (a0, a1) in zip(actions_0, actions_1)]
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, next_info = envs.step(action)
            #next_obs, reward, done, next_info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
#            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in next_info: #one per env in envs
                if "episode" in item.keys(): #"episode" metrics are automatically returned by the env wrapper RecordEpisodeStatistics when an episode ends
                    print(f"global_step={global_step}, water={item['metrics']['water_produced']} lichen={item['metrics']['lichen_produced']}")
                    writer.add_scalar("charts/episodic_return",                         item["episode"]["r"],                           global_step)
                    writer.add_scalar("charts/episodic_length",                         item["episode"]["l"],                           global_step)
                    writer.add_scalar('charts/episodic_lichen_produced',                item["metrics"]["lichen_produced"],             global_step)
                    writer.add_scalar('charts/episodic_water_produced',                 item["metrics"]["water_produced"],              global_step)
                    writer.add_scalar('charts/episodic_ice_dug',                        item["metrics"]["ice_dug"],                     global_step)
                    writer.add_scalar('charts/episodic_action_queue_updates_success',   item["metrics"]["action_queue_updates_success"],global_step)
                    writer.add_scalar('charts/episodic_action_queue_updates_total',     item["metrics"]["action_queue_updates_total"],  global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            
            next_obs_0 = torch.stack([torch.Tensor(o[0][0]) for o in next_obs]).to(device)  #use only the first unit of player_0 for training, collected across all parallel envs
            next_value = agent.get_value(next_obs_0).reshape(1, -1)  #use only first unit
            #next_value = agent.get_value(next_obs_0).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    
    torch.save(agent.state_dict(), f'models/{run_name}')
    print(f'Run Name: {run_name}')
    
    
    