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

from wrappers import SimpleUnitDiscreteController_3


#Each factory in Agent_3_1 will first create a Heavy unit, and then as many Light units as resources permin
#each unit is tied to the factory that created it; all units are controlled with the same model

#should be the same NN as in ppo_train_3_1.py
MODEL_WEIGHTS_RELATIVE_PATH = "models/LuxAI_S2-v0__ppo_train_3_1__1__1679000427"

SHAPE = 13-3-1-1-2+9+4

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

#same as in ppo_train_3_1
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


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.agent = AgentModel(obs_dim=SHAPE, action_dim=12)  #added water cargo of mother factory to unit's observation space as in train_1
        self.agent.load_state_dict(torch.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH)))

        self.controller = SimpleUnitDiscreteController_3(self.env_cfg)
        self.units_factories = {'player_0':{}, 'player_1':{}}  #to remember the relation between units and their mother factories
        

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

#        units_ids, obs_units = self.obs_for_units(obs) 

        #process the general observation data into an array of observation vectors - one per unit - to pass to the NN model to control units
        obs_units = obs_for_units(raw_obs, self.player, self.units_factories, self.env_cfg)
        
        action_masks = self.controller.action_masks(self.player, raw_obs, multi_agent=True) #assume action_masks are in same order

        if len(obs_units)>0:
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
                
                
            #units_actions = {unit_id:action for _, (unit_id, action) in enumerate(zip(units_ids, actions))}
            lux_action = self.controller.actions_to_lux_actions(self.player, raw_obs, actions)

        ####################
        ## factories actions
        ####################

        factory_actions = self.controller.factory_actions(self.player, raw_obs)
        lux_action.update(factory_actions)
        
        '''
        #create units
        factories = obs["factories"][self.player]
        for _, (factory_id, factory) in enumerate(factories.items()):
            if factory_id in self.units_factories.values(): #does this factory already have units attached to it
                lux_action[factory_id] = 0  # yes, the factory tries to build a light unit (it has already built a heavy one)
            else:
                lux_action[factory_id] = 1  # no, then factory first tries to build a heavy unit 
            
            
        #if len(units_ids) == 0:
        #    for factory_id in factories.keys():
        #        lux_action[factory_id] = 1  # each factory builds a heavy unit
    


        #watering lichen which can easily improve your agent
        #shared_obs = raw_obs[self.player]
        #factories = shared_obs["factories"][self.player]
        for factory_id in factories.keys():
            factory = factories[factory_id]
            if 1000 - step < 50 and factory["cargo"]["water"] > 100:
                lux_action[factory_id] = 2 # water and grow lichen at the very end of the game
        '''
        return lux_action
