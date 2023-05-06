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



####################################################
MODEL_WEIGHTS_RELATIVE_PATH = "models/LuxAI_S2-v0__train__1__05-Apr-2023-22.13.26"
ACTION_SHAPE = 12
####################################################
#prepare a list of observation vectors, one for each unit, that can be used by the NN to find the best action for each unit

#length of the observation vector for a single robot;

INDEX = 3  #player#,type#,unit#

SCALAR = 0
SCALAR += 2+5+1 #pos, cargo,type
SCALAR += 2+1  #factory pos, water

ICE_IMG = 25+9+9
RUBBLE_IMG = 25+9+9
UNITS_IMG = 25+9+9

SHAPE = 0
SHAPE += SCALAR
SHAPE += ICE_IMG
SHAPE += RUBBLE_IMG
SHAPE += UNITS_IMG

ICE_FEATURES = 8
RUBBLE_FEATURES = 8
UNITS_FEATURES = 8
COMBINED_NEURONS = 16
CRITIC_NEURONS = 32
ACTOR_NEURONS = 32

def simple_convolution( agent, img, scale ):
    
        padded_img = np.pad(img,(2,2)) #pad the matrix to facilitate the lookup of 5x5 neighborhoods
        x, y = agent[0], agent[1] #top left corner of 5x5 neighborhood in padded matrix
        neighborhood = padded_img[np.ix_([y+0,y+1,y+2,y+3,y+4],[x+0,x+1,x+2,x+3,x+4])].flatten()  
        
        quadrants = np.zeros(9)
        x, y = agent[0], agent[1] #top left corner of the north east 5x5 quadrant
        padded_img = np.pad(img,(2+5,2+5)) #pad the matrix to facilitate the lookup of 5x5 quadrants around the 5x5 immediate neighborhood
        for i in range(3):
            for j in range(3):
                quadrants[j+3*i] = padded_img[y:y+5, x:x+5].sum() / scale
                x += 5 
            y += 5
            x = agent[0]

        sectors = np.zeros(9)
        padded_img = np.pad(img,(2+5+15,2+5+15)) #pad the matrix to facilitate the lookup of 5x5 quadrants around the 5x5 immediate neighborhood
        x, y = agent[0], agent[1] #top left corner of sector in padded matrix
        for i in range(3):
            for j in range(3):
                sectors[j+3*i] = padded_img[y:y+15, x:x+15].sum() / scale
                x += 15 
            y += 15
            x = agent[0]

        return neighborhood, quadrants, sectors

def obs_for_units(obs, player, units_factories, env_cfg):

    shared_obs = obs['player_0']
    ice_map = shared_obs["board"]["ice"]
    rubble_map = shared_obs["board"]["rubble"]
#    ore_map = shared_obs["board"]["ore"]


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
    if len(units_list)>0:
        units_map[units_list[:,0], units_list[:,1]] = units_list[:,2]  

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
        neighborhood_ice, quadrants_ice, sectors_ice = simple_convolution(unit['pos'], img=ice_map, scale=10)
        neighborhood_rubble, quadrants_rubble, sectors_rubble = simple_convolution(unit['pos'], img=rubble_map, scale=10)
        #neighborhood_ore, quadrants_ore, sectors_ore = simple_convolution(unit['pos'], img=ore_map, scale=10)
        neighborhood_units, quadrants_units, sectors_units = simple_convolution(unit['pos'], img=units_map, scale=10)

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

        '''
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
        '''
        
        #get the water stock of the mother factory as an additional observation for this unit, normalized to a reasonable scale
        #this is the same logic as as SimpleUnitObservationWrapper_1
        factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
        
        #put the entire observation together
#            obs_vec = np.concatenate([unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1)
        obs_vec = np.concatenate([unit_vec, 
                                  factory_vec - pos, 
                                  factory_water,
                                  neighborhood_ice, quadrants_ice, sectors_ice,
#                                  neighborhood_ore, quadrants_ore, sectors_ore,
                                  neighborhood_rubble, quadrants_rubble, sectors_rubble,
                                  neighborhood_units, quadrants_units, sectors_units
                                  ], axis=-1)  
        
        #return two arrays ordered in the same way, with units keys and their corresponding observations
        units_ids.append(unit['unit_id'])  
        observations.append(obs_vec)

    '''
    NOT NEEDED FOR REPLAYING
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
    '''
    
    return observations, units_ids


####################################################
####################################################
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentNN(nn.Module):
    def __init__(self, action_shape):
        super().__init__()
        
        self.ice_features = nn.Sequential(
            layer_init(nn.Linear(ICE_IMG, ICE_FEATURES)),
            nn.ReLU(),
            layer_init(nn.Linear(ICE_FEATURES, ICE_FEATURES)),
            nn.ReLU()
        )        

        self.rubble_features = nn.Sequential(
            layer_init(nn.Linear(RUBBLE_IMG, RUBBLE_FEATURES)),
            nn.ReLU(),
            layer_init(nn.Linear(RUBBLE_FEATURES, RUBBLE_FEATURES)),
            nn.ReLU()
        )

        self.units_features = nn.Sequential(
            layer_init(nn.Linear(UNITS_IMG, UNITS_FEATURES)),
            nn.ReLU(),
            layer_init(nn.Linear(UNITS_FEATURES, UNITS_FEATURES)),
            nn.ReLU()
        )

        #self.combined = nn.Sequential(
        #    layer_init(nn.Linear(SCALAR + ICE_FEATURES + RUBBLE_FEATURES + UNITS_FEATURES, COMBINED_NEURONS)),
        #    nn.ReLU(),
        #    layer_init(nn.Linear(COMBINED_NEURONS, COMBINED_NEURONS)),
        #    nn.ReLU()
        #)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(SCALAR + ICE_FEATURES + RUBBLE_FEATURES + UNITS_FEATURES, CRITIC_NEURONS)),
            nn.Tanh(),
            layer_init(nn.Linear(CRITIC_NEURONS, CRITIC_NEURONS)),
            nn.Tanh(),
            layer_init(nn.Linear(CRITIC_NEURONS, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(SCALAR + ICE_FEATURES + RUBBLE_FEATURES + UNITS_FEATURES, ACTOR_NEURONS)),
            nn.Tanh(),
            layer_init(nn.Linear(ACTOR_NEURONS, ACTOR_NEURONS)),
            nn.Tanh(),
            layer_init(nn.Linear(ACTOR_NEURONS, action_shape), std=0.01),
        )

    def get_combined(self, x):
        
        ice_i = x[:,SCALAR:SCALAR+ICE_IMG]
        ice_o = self.ice_features(ice_i)
        
        rubble_i = x[:,SCALAR+ICE_IMG:SCALAR+ICE_IMG+RUBBLE_IMG]
        rubble_o = self.rubble_features(rubble_i)
        
        units_i = x[:,SCALAR+ICE_IMG+RUBBLE_IMG:SCALAR+ICE_IMG+RUBBLE_IMG+UNITS_IMG]
        units_o = self.units_features(units_i)

        c_i = torch.cat([x[:,:SCALAR], ice_o, rubble_o, units_o], dim=1)
        #c_o = self.combined(c_i)
        
        return c_i

    def get_value(self, x):
        c_i = self.get_combined( x ) 
        return self.critic( c_i )

    def get_action(self, x):
        c_i = self.get_combined( x )  
        logits = self.actor( c_i )
        return logits
       

    def get_action_and_value(self, x, action=None):
        c_i = self.get_combined( x )  
        logits = self.actor( c_i )
        
#        logits = torch.where(invalid_action_masks==1., logits, torch.tensor(-1e+8))
           
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic( c_i )

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)

        self.heavy = AgentNN(ACTION_SHAPE)  #added water cargo of mother factory to unit's observation space as in train_1
        self.heavy.load_state_dict(torch.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH + '_heavy')))

        self.light = AgentNN(ACTION_SHAPE)  #added water cargo of mother factory to unit's observation space as in train_1
        self.light.load_state_dict(torch.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH + '_light')))

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

        lux_action = {}
        
        ##############
        #units actions
        ##############

        action_masks = self.controller.action_masks(self.player, raw_obs, multi_agent=True) #assume action_masks are in same order
        action_masks = torch.Tensor(np.array(action_masks)).bool()

        #process the general observation data into an array of observation vectors - one per unit - to pass to the NN model to control units
        obs_units, units_ids = obs_for_units(raw_obs, self.player, self.units_factories, self.env_cfg)
        obs_units = np.array(obs_units)
        units_ids = np.array(units_ids)

        if len(obs_units)>0:
            heavies = obs_units[obs_units[:,1]==1]  #select only heavy units; first 3 columns of flat_obs are: player, unit-type, unit-id
            heavies_ids = units_ids[obs_units[:,1]==1]
            heavies_masks = action_masks[obs_units[:,1]==1]
    
            lights = obs_units[obs_units[:,1]==0]  #select only light units; first 3 columns of flat_obs are: player, unit-type, unit-id
            lights_ids = units_ids[obs_units[:,1]==0]
            lights_masks = action_masks[obs_units[:,1]==0]
    
            action={}
            
            #run the NN for heavy units
            if len(heavies)>0:
                with torch.no_grad():
                    logits = self.heavy.get_action(torch.from_numpy(np.array(heavies[:,INDEX:])).float()) #return shape (num heavy units, num possible actions) 
                    logits[~heavies_masks] = -1e8 # mask out invalid actions
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample().cpu().numpy() # shape (num heavy units, 1)
    
                for _,(k,a) in enumerate(zip(heavies_ids, actions)):
                        action[k]=a
    
            #run the NN for light units
            if len(lights)>0:
                with torch.no_grad():
                    logits = self.light.get_action(torch.from_numpy(np.array(lights[:,INDEX:])).float()) #return shape (num heavy units, num possible actions) 
                    logits[~lights_masks] = -1e8 # mask out invalid actions
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample().cpu().numpy() # shape (num heavy units, 1)
    
                for _,(k,a) in enumerate(zip(lights_ids, actions)):
                        action[k]=a
    
            #units_actions = {unit_id:action for _, (unit_id, action) in enumerate(zip(units_ids, actions))}
            lux_action = self.controller.actions_to_lux_actions(self.player, raw_obs, action)


        ####################
        ## factories actions
        ####################

        factory_actions = self.controller.factory_actions(self.player, raw_obs)
        lux_action.update(factory_actions)
        
        return lux_action
