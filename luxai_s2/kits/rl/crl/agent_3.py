"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


#from stable_baselines3.dqn import DQN
#from lux.config import EnvConfig
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory

from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper#_3


#should be the same NN as in ppo_train.py

MODEL_WEIGHTS_RELATIVE_PATH = "models/LuxAI_S2-v0__ppo_train_3__1__1678368751"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AgentModel(nn.Module):  
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )




class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        #self.agent = AgentModel(obs_dim=13+1, action_dim=12)
        self.agent = AgentModel(obs_dim=13, action_dim=12)
        self.agent.load_state_dict(torch.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH)))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)
        
        self.units_factories = {} #to remember the relation between units and their mother factories

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        faction = 'AlphaStrike'
        if self.player == 'player_1':
            faction = 'MotherMars'
        return dict(faction=faction, bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving

        #avoid sending an action out of turn
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories_to_place = game_state.teams[self.player].factories_to_place
        # whether it is your turn to place a factory
        my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)

        if not my_turn_to_place or factories_to_place==0:
            return dict()

        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"] / factories_to_place #should be 150
        water = obs["teams"][self.player]["water"] / factories_to_place #should be 150
        return dict(spawn=pos, metal=metal, water=water)

    #wrapper to prepare a list of observation vectors, one for each heavy unit, that can be used by the NN to find the best action for each unit
    def obs_for_heavies(self, obs):

        shared_obs = obs
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        factories = shared_obs['factories'][self.player]
        factories_locations = np.array([ np.array(factories[factory_id]['pos']) for factory_id in factories.keys()])

        units_ids = []    #return the list of units, to make sure we preserve order    
        observations = [] #return an array of observations, one for each unit
        
        units = shared_obs["units"][self.player]
        for k in units.keys():
            unit = units[k]

            obs_vec = np.zeros(13+1) #size of observation space for NN model
                
            # store cargo+power values scaled to [0, 1]
            cargo_space = self.env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
            battery_cap = self.env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
            cargo_vec = np.array(
                [
                    unit["power"] / battery_cap,
                    unit["cargo"]["ice"] / cargo_space,
                    unit["cargo"]["ore"] / cargo_space,
                    unit["cargo"]["water"] / cargo_space,
                    unit["cargo"]["metal"] / cargo_space,
                ]
            )

            unit_type = (
                0 if unit["unit_type"] == "LIGHT" else 1
            )  # note that build actions use 0 to encode Light

            # normalize the unit position
            pos = np.array(unit["pos"]) / self.env_cfg.map_size
            
            unit_vec = np.concatenate(
                [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
             )

            # we add some engineered features down here

            # compute closest ice tile
            ice_tile_distances = np.mean(
                (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
            )
            closest_ice_tile = (
                ice_tile_locations[np.argmin(ice_tile_distances)] / self.env_cfg.map_size
            )

            '''
            '''
            #tell the unit to focus on its mother factory
            if (k not in self.units_factories) \
                or (self.units_factories[k] not in factories):  
                #the unit has just been created and is not yet associated with a factory
                #or the mother factory has died and we need to focus on a different one
                
                # identify the closest factory
                factories_distances = np.mean(
                    (factories_locations - np.array(unit["pos"])) ** 2, 1
                )
                i = np.argmin(factories_distances)
                factory_id = list(factories)[i]  #rely on the fact that dictionaries are order-preserving
                self.units_factories[k]=factory_id
            
            #we want the unit to focus on this specific factory and not wander off
            factory_id = self.units_factories[k]
            factory = factories[factory_id] 
            factory_vec = factory['pos']/self.env_cfg.map_size
            factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
            
            #put the entire observation together
            obs_vec = np.concatenate(
                [unit_vec, factory_vec - pos, closest_ice_tile - pos, factory_water], axis=-1
            )
            
            #return two arrays ordered in the same way, with units keys and their corresponding observations
            units_ids.append(unit['unit_id'])  
            observations.append(obs_vec)

        return units_ids, observations


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
#        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
#        obs = obs[self.player]

        lux_action = {}
        
        ##############
        #units actions
        ##############

        units_ids, obs_units = self.obs_for_heavies(obs) #process the general observation data into an array of observation vectors - one per unit - to pass to the NN model for HEAVY units
        
        action_masks = self.controller.action_masks(self.player, raw_obs, multi_agent=True) #assume action_masks are in same order

        if len(units_ids)>0:
            with torch.no_grad():
    
                # to improve performance, we have a rule based action mask generator for the controller used
                # which will force the agent to generate actions that are valid only.
                #action_mask = (
                #    torch.from_numpy(self.controller.action_masks(self.player, raw_obs))
                #    .unsqueeze(0)
                #    .bool()
                #)

               
                logits = self.agent.actor(torch.from_numpy(np.array(obs_units)).float()) #return shape (num heavy units, num possible actions) 
                
                action_masks = (
                    torch.from_numpy(np.array(action_masks))
                    .bool()
                )
                
                logits[~action_masks] = -1e8 # mask out invalid actions
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().cpu().numpy() # shape (num heavy units, 1)
                
                
            # use our controller which we trained with in train.py to generate a Lux S2 compatible action
            units_actions = {unit_id:action for _, (unit_id, action) in enumerate(zip(units_ids, actions))}
            lux_action = self.controller.actions_to_lux_actions(self.player, raw_obs, units_actions)

        ####################
        ## factories actions
        ####################
        
        #create units
        factories = obs["factories"][self.player]
        
        if len(units_ids) == 0:
            for factory_id in factories.keys():
                lux_action[factory_id] = 1  # each factory builds a heavy unit

        #watering lichen which can easily improve your agent
        #shared_obs = raw_obs[self.player]
        #factories = shared_obs["factories"][self.player]
        for factory_id in factories.keys():
            factory = factories[factory_id]
            if 1000 - step < 50 and factory["cargo"]["water"] > 100:
                lux_action[factory_id] = 2 # water and grow lichen at the very end of the game

        return lux_action