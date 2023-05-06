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
#-1 for team id
#-2 for closest ice tile
# +9 for ice neighborhood i the 3x3 area centered on the unit
# +4 for ice in the 4 quadrants (north, south, east, west)
# +9 for units in 3x3 neighborhood TODO: could remove center...
# +9 for ore in 3x3 neighborhood TODO
# +4 for ore in the 4 quadrants (north, south, east, west)
# +2 for what is on unit's tile (rubble or lichen)

SHAPE = 13-1-2+9+4+9+9+4+2
INDEX = 3  #player#,type#,unit#

def obs_for_units(obs, player, units_factories, env_cfg):

    shared_obs = obs['player_0']
    ice_map = shared_obs["board"]["ice"]
    ice_tile_locations = np.argwhere(ice_map == 1)
    padded_ice_map = np.pad(ice_map,(1,1)) #pad the matrix to facilitate the lookup of 3x3 neighborhoods
    total_ice = np.count_nonzero(ice_tile_locations) #count of ice on the board, for normalizing

    ore_map = shared_obs["board"]["ore"]
    padded_ore_map = np.pad(ore_map,(1,1)) #pad the matrix to facilitate the lookup of 3x3 neighborhoods
    ore_tile_locations = np.argwhere(ore_map == 1)
    total_ore = np.count_nonzero(ore_tile_locations) #count of ice on the board, for normalizing

    factories = shared_obs['factories'][player]
    factories_locations = np.array([ np.array(factories[factory_id]['pos']) for factory_id in factories.keys()])

    unit_sign = 1 if player == 'player_0' else -1
    
    units_list = [ 
        [u['pos'][0], u['pos'][1], unit_sign *(10 if u['unit_type']=='HEAVY' else 1)] 
        for u in shared_obs['units']['player_0'].values()
    ]
    
    units_list += [
        [u['pos'][0], u['pos'][1], - unit_sign *(10 if u['unit_type']=='HEAVY' else 1)] 
        for u in shared_obs['units']['player_1'].values()
    ]
    
    units_list = np.array(units_list)
    units_map = np.zeros((env_cfg.map_size,env_cfg.map_size))
    units_map[units_list[:,0], units_list[:,1]] = units_list[:,2]  
    padded_units_map = np.pad(units_map,(1,1))

    units_ids = []    #return the list of units, to make sure we preserve order    
    observations = [] #return an array of observations, one for each unit
    
    index = [0 if player=='player_0' else 1, 0, 0]  #meta data with player, unit type and unit id, should  be length=INDEX
    
    
    found_light = False
    found_heavy = False
    
    units = shared_obs["units"][player]
    for i,(k,unit) in enumerate(units.items()):

        index[1] = 0 if unit["unit_type"]=="LIGHT" else 1
        if index[1]==0: found_light=True
        if index[1]==1: found_heavy=True
        index[2] = i
        
        obs_vec = np.zeros(SHAPE) #size of observation space for NN model
            
        # store cargo+power values scaled to [0, 1]
        cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
        battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
        cargo_vec = np.array(
            [
                unit["power"] / battery_cap,
                unit["cargo"]["ice"] / cargo_space,
                unit["cargo"]["ore"] / cargo_space,
                unit["cargo"]["water"] / cargo_space,
                unit["cargo"]["metal"] / cargo_space 
            ]
        )

        unit_type = (-1 if unit["unit_type"] == "LIGHT" else 1)

        # normalize the unit position
        pos = np.array(unit["pos"]) / env_cfg.map_size
        
        unit_vec = np.concatenate([index, pos, cargo_vec, [unit_type]], axis=-1)
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

        #add the observation of other close-by units
        neighborhood_units = padded_units_map[np.ix_([x+1,x+2,x+3],[y+1,y+2,y+3])].flatten()  

        #add the ore inventory
        neighborhood_ore = padded_ore_map[np.ix_([x+1,x+2,x+3],[y+1,y+2,y+3])].flatten()  
        north_ore = ore_map[:,:y].sum()  #total ice north of the unit's immediate neighborhood
        south_ore = ore_map[:,y+3:].sum()
        west_ore = ore_map[:x,:].sum()  
        east_ore = ore_map[x+3:,:].sum() 
        quadrants_ore = np.array([north_ore, south_ore, east_ore, west_ore])/total_ore #normalize with a reasonable scale


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

        #tell the unit if it should dig for rubble
        #rubble is bad when near my factory (costs power, and prevents lichen)
        #but it is good when near the opponent factory, i.e. far
        is_rubble = shared_obs["board"]["ice"][unit['pos'][0]][unit['pos'][1]]/100
        
        is_lichen = shared_obs["board"]["lichen_strains"][unit['pos'][0]][unit['pos'][1]]
        if is_lichen==-1:
            is_lichen=0
        elif is_lichen in shared_obs['teams'][player]['factory_strains']:
            is_lichen=1 #this is good lichen, it belongs to a factory in my team
        else:
            is_lichen=-1 #this is bad lichen, it belongs to the other guy
        
        #get the water stock of the mother factory as an additional observation for this unit, normalized to a reasonable scale
        #this is the same logic as as SimpleUnitObservationWrapper_1
        #factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
        
        #put the entire observation together
#            obs_vec = np.concatenate([unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1)
        obs_vec = np.concatenate([unit_vec, factory_vec - pos, neighborhood_ice, quadrants_ice, neighborhood_ore, quadrants_ore, neighborhood_units/10.,[is_rubble, is_lichen]], axis=-1)  #scale units by 10, to bring heavies back to 1
        
        #return two arrays ordered in the same way, with units keys and their corresponding observations
        units_ids.append(unit['unit_id'])  
        observations.append(obs_vec)

    if not found_light:
        obs_vec = np.zeros(INDEX+SHAPE)
        obs_vec[0] = 0 if player=='player_0' else 1
        obs_vec[1] = 0  #mark it as a light unit
        obs_vec[2] = -1 #mark it as fake unit
        observations.append(obs_vec)
        units_ids.append('')

    if not found_heavy: 
        obs_vec = np.zeros(INDEX+SHAPE)
        obs_vec[0] = 0 if player=='player_0' else 1
        obs_vec[1] = 1  #mark it as a heavy unit
        obs_vec[2] = -1 #mark it as fake unit
        observations.append(obs_vec)
        units_ids.append('')

    return observations, units_ids



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

        self.observations = None  #array of NN-compatible observation vectors, player is axis zero, unit is axis one, obs vector for each unit is axis 3
        self.units_ids = []
        
        self.units_factories = {'player_0':{}, 'player_1':{}}  #to remember the relation between units and their mother factories

    def observation(self):
        return {'vectors': self.observations, 'units_ids':self.units_ids}

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
        done = done[agent]

        #note: we need to guarantee that the observation returned by this wrapper will always have obs[0][0] and obs[1][]: this is needed for training
        if not done:
            obs_1, units_ids_1 = obs_for_units(obs, 'player_0', self.units_factories['player_0'], self.env.env_cfg)
            obs_2, units_ids_2 = obs_for_units(obs, 'player_1', self.units_factories['player_1'], self.env.env_cfg)
            self.observations = np.concatenate((obs_1,obs_2))
            self.units_ids = units_ids_1 + units_ids_2

        
        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        metrics["ore_dug"] = stats["generation"]["ore"]['HEAVY'] + stats["generation"]["ore"]['LIGHT']        
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["metal_produced"] = stats["generation"]["metal"]
        metrics["lichen_produced"] = stats["generation"]["lichen"]
        metrics["units"] = stats["generation"]["built"]["HEAVY"] + stats["generation"]["built"]["LIGHT"]

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
            ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
            metal_produced_this_step = metrics["metal_produced"] - self.prev_step_metrics["metal_produced"]
            water_produced_this_step = metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            lichen_produced_this_step = metrics["lichen_produced"] - self.prev_step_metrics["lichen_produced"]

            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 
            + 1 * water_produced_this_step 
            + 1 * ore_dug_this_step 
            + 10 * metal_produced_this_step 
            + 100 * lichen_produced_this_step

        self.prev_step_metrics = copy.deepcopy(metrics)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):

        self.units_factories = {'player_0':{}, 'player_1':{}}  #to remember the relation between units and their mother factories

        obs = self.env.reset(**kwargs)    #this is calling SinglePhaseWrapper() and will go through the factory placement process
        
        self.prev_step_metrics = None

        #run the first step after factory placement, to create the first units before training can properly start
        
        action = {'player_0':{}, 'player_1':{}}       
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
    
    parser.add_argument("--total-timesteps", type=int, default=20_000_000,
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


class Trainee():
    
    def __init__(self, envs, args, run_name, model_name, model_filter):
        
        self.envs = envs
        self.args = args
        self.run_name = run_name
        self.model_name = model_name
        self.model_filter = model_filter

        self.writer = SummaryWriter(f"../runs/{run_name}_{model_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        
        self. agent = AgentNN(envs).to(device)    #used for training
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    
    def run_step(self, flat_obs, units_ids):
        
        #flat_obs shoudl be a flat array of observation vectors, one entry per unit
        #columns should be
        #0: the index of the parallel env
        #1: the player
        #2: the type of unit (0:LIGHT, 1:HEAVY)
        #3: the index of the unit
        
        #units_ids should be an array in the same order, with the unit ids strings that are recognized by luxai_s2' env
        
        heavies = flat_obs[flat_obs[:,2]==self.model_filter]  #select only heavy units; first 4 columns of flat_obs are: env, player, unit-type, unit-id
        keys = units_ids[flat_obs[:,2]==self.model_filter] #matching keys of the selected units
        
        with torch.no_grad():
            
            #run the model for all heavy units across all environments and players
            
            o = heavies[:,1+INDEX:] #ignore the meta data, keep only what the NN expects
            action, logprob, _, value = self.agent.get_action_and_value(o)

            obs_ = []
            values_ = []
            actions_ = []
            logprobs_ = []
            actions = []
            
            #now we need to regroup by parallel env

            for i in range(self.args.num_envs): #iterate on the parallel envs
            
                #select one unit to record for training
                
                e = heavies[:,0]==i #select the env
                player_0 =  heavies[e][:,1]==0 #select player_0 in this env
                player_1 =  heavies[e][:,1]==1 #select player_1 in this env
                selected = np.random.choice(np.count_nonzero(player_0), size=1) #select one unit from player_0 of this env

                obs_.append( heavies[e][player_0][selected][:,1+INDEX:].squeeze() )
                values_.append( value[e][player_0][selected].squeeze() )
                actions_.append( action[e][player_0][selected].squeeze() )
                logprobs_.append( logprob[e][player_0][selected].squeeze() )

                #format the actions for all units of this type
                
                action_0={}
                for _,(k,a) in enumerate(zip(keys[e][player_0], action[e][player_0])):
                    if k != '':
                        action_0[k]=a

                action_1={}
                for _,(k,a) in enumerate(zip(keys[e][player_1], action[e][player_1])):
                    if k != '':
                        action_1[k]=a
                
                action_ = {'player_0': action_0, 'player_1': action_1}
                actions.append(action_)

            #store the selected obs in the buffer for training                        
            
            self.obs[step] = torch.stack(obs_)
            self.values[step] = torch.stack(values_)
            self.actions[step] = torch.stack(actions_)
            self.logprobs[step] = torch.stack(logprobs_)
            self.dones[step] = next_done

        return actions #note: these actions are only for the units of the selected model

    def pick_one_obs(self, next_obs):
        next_obs_0 = []
        for i, o in enumerate(next_obs): #one set of observations per parallel env
            #select one unit of the right kind per parallel env to record for training
            v = o['vectors']  #one table of vectors, for each unit of the env (both players, both types of units)
            player_0 = v[v[:,0]==0] #select the units that belong to the first player
            units = player_0[player_0[:,1]==self.model_filter]  #select the right kind of units; first 3 columns of : player, unit-type, unit-id
            selected = np.random.choice(units.shape[0], size=1) #select one unit from player_0 of this env
            next_obs_0.append( units[selected][:,INDEX:].squeeze() )
        
        next_obs_0 = torch.Tensor(np.array(next_obs_0)).to(device)
        return next_obs_0 #one observation per environment
        

    def train_step(self, next_obs, next_done, update, num_updates, global_step):
        
        # Annealing the rate if instructed to do so.
        if self.args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * self.args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        
        # bootstrap value if not done
        with torch.no_grad():
            
            #pick one unit of the right kind per environment, and determine the best action from the agent's policy
            next_obs_0 = self.pick_one_obs(next_obs)
            next_value = self.agent.get_value(next_obs_0).reshape(1, -1)  
            
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    
  
    

####################################################
####################################################
if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = MySyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.max_episode_steps) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # agent setup
    heavy = Trainee(envs, args, run_name, model_name="heavy", model_filter=1)
    light = Trainee(envs, args, run_name, model_name="light", model_filter=0)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates + 1):

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: action logic

            #the idea is to call the NN on all units of the same types across all envs, instead of calling it separately for each env
            #flatten the parallel env's observations into a single array of observation vectors and the corresponding units' ids.
            #add a column to the obs vectors, to remember which env they belong to
            
            flat_obs = []
            units_ids = []
            for i,o in enumerate(next_obs): #one set of observations per parallel env
                #add a column with the env number
                v = o['vectors']  #one table of vectors, for each unit of the env (both players, both types of units)
                k = o['units_ids'] #corresponding keys of the units, in same order
                o1 = np.hstack( (i*np.ones(v.shape[0]).reshape(-1,1), v))  #add a column (index=0) to remember the parallel env
                flat_obs.append(o1)
                units_ids += k
            flat_obs = torch.Tensor(np.concatenate(flat_obs)).to(device)
            units_ids = np.array(units_ids) #same order as in flat_obs

            # run the models for each type of unit
            
            heavy_actions = heavy.run_step(flat_obs, units_ids) #determine what actions should be taken for all heavy units for all envs and players
            
            light_actions = light.run_step(flat_obs, units_ids) #determine what actions should be taken for all light units for all envs and players
            
            actions = [] #one entry per parallel env, combining actions for lights and heavy units for both players
            for _,(h,l) in enumerate(zip(heavy_actions, light_actions)):
                a = h
                a['player_0'].update(l['player_0'])
                a['player_1'].update(l['player_1'])
                actions.append(a)
                
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, next_info = envs.step(actions)
            heavy.rewards[step] = torch.tensor(reward).to(device).view(-1)
            light.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(done).to(device)

            for item in next_info: #one per env in envs
                if "episode" in item.keys(): #"episode" metrics are automatically returned by the env wrapper RecordEpisodeStatistics when an episode ends
                    print(f"global_step={global_step}, water={item['metrics']['water_produced']} metal={item['metrics']['metal_produced']} lichen={item['metrics']['lichen_produced']}")
                    heavy.writer.add_scalar("charts/episodic_return",                         item["episode"]["r"],                           global_step)
                    heavy.writer.add_scalar("charts/episodic_length",                         item["episode"]["l"],                           global_step)
                    heavy.writer.add_scalar('charts/episodic_lichen_produced',                item["metrics"]["lichen_produced"],             global_step)
                    heavy.writer.add_scalar('charts/episodic_ore_dug',                        item["metrics"]["ore_dug"],                     global_step)
                    heavy.writer.add_scalar('charts/episodic_metal_produced',                 item["metrics"]["metal_produced"],              global_step)
                    heavy.writer.add_scalar('charts/episodic_water_produced',                 item["metrics"]["water_produced"],              global_step)
                    heavy.writer.add_scalar('charts/episodic_ice_dug',                        item["metrics"]["ice_dug"],                     global_step)
                    heavy.writer.add_scalar('charts/episodic_units',                          item["metrics"]["units"],                       global_step)
                    heavy.writer.add_scalar('charts/episodic_action_queue_updates_success',   item["metrics"]["action_queue_updates_success"],global_step)
                    heavy.writer.add_scalar('charts/episodic_action_queue_updates_total',     item["metrics"]["action_queue_updates_total"],  global_step)
                    break
            
            #end for step in range(0, args.num_steps)
        
        #update the models once the batch of observations has been gathered
        heavy.train_step(next_obs, next_done, update, num_updates, global_step)
        light.train_step(next_obs, next_done, update, num_updates, global_step)
        
    envs.close()
    heavy.writer.close()
    
    torch.save(heavy.agent.state_dict(), f'models/{run_name}_heavy')
    torch.save(light.agent.state_dict(), f'models/{run_name}_light')
    print(f'Run Name: {run_name}')
    
    
    