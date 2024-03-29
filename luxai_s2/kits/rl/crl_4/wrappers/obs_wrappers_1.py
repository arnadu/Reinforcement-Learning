from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces


class SimpleUnitObservationWrapper_1_2(gym.ObservationWrapper):
    """
    Returns an observation vector for the first unit only
    Included features
    - robot's stats
    - distance vector to closest ice tile
    - distance vector to mother factory (the first factory in the list)
    
    - neighborhood: ice
    
    """
    
    #length of the observation vector for a single robot;
    #13 is the size of the original example code
    #-3 for cargo (all cargo types summed up into a single number instead)
    #-1 for unit type
    #-1 for team id
    #-2 for closest ice tile
    # +9 for ice neighborhood i the 3x3 area centered on the unit
    # +4 for ice in the 4 quadrants (north, south, east, west)
    
    SHAPE = 13-3-1-1-2+9+4
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(SimpleUnitObservationWrapper_1_2.SHAPE,)) #one observation vector per robot unit, with added observation of mother factory's water to the units vector

    def observation(self, obs):
        return SimpleUnitObservationWrapper_1_2.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        padded_ice_map = np.pad(ice_map,(1,1)) #pad the matrix to facilitate the lookup of 3x3 neighborhoods
        ice_tile_locations = np.argwhere(ice_map == 1)
        total_ice = np.count_nonzero(ice_tile_locations) #count of ice on the board, for normalizing

        for agent in obs.keys():
            obs_vec = np.zeros(SimpleUnitObservationWrapper_1_2.SHAPE) #in case there is no unit, still need to return an obs of the right shape

            factories = shared_obs["factories"][agent]
            #factory_water = np.zeros(1)  #define variables, in case there is no factory left
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                #get the water stock of the mother factory as an additional observation for this unit, normalized to a reasonable scale
                #factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
                break #returns info about the first factory only
            
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

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
#                        unit["cargo"]["ore"] / cargo_space,
#                        unit["cargo"]["water"] / cargo_space,
#                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
#                unit_type = (
#                    0 if unit["unit_type"] == "LIGHT" else 1
#                )  # note that build actions use 0 to encode Light
                # normalize the unit position

                pos = np.array(unit["pos"]) / env_cfg.map_size

#                unit_vec = np.concatenate([pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1)
                unit_vec = np.concatenate([pos, cargo_vec], axis=-1)

                # we add some engineered features down here
                # compute closest ice tile
                
                #ice_tile_distances = np.mean((ice_tile_locations - np.array(unit["pos"])) ** 2, 1)
                #closest_ice_tile = (ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size)
                
                #get the ice inventory in the immediate neighborhood, and the 4 quadrants
                x, y = unit['pos'][0]-1, unit['pos'][1]-1 #top left corner of the 3x3 neighborhood centered on the unit
                
                neighborhood_ice = padded_ice_map[np.ix_([x+1,x+2,x+3],[y+1,y+2,y+3])].flatten()  
                
                north_ice = ice_map[:,:y].sum()  #total ice north of the unit's immediate neighborhood
                south_ice = ice_map[:,y+3:].sum()
                west_ice = ice_map[:x,:].sum()  
                east_ice = ice_map[x+3:,:].sum() 
                quadrants_ice = np.array([north_ice, south_ice, east_ice, west_ice])/total_ice #normalize with a reasonable scale
                
                #obs_vec = np.concatenate([unit_vec, factory_vec - pos, closest_ice_tile - pos, neighborhood_ice], sectors_ice, axis=-1)
                obs_vec = np.concatenate([unit_vec, factory_vec - pos, neighborhood_ice, quadrants_ice], axis=-1)
                
                break #return info about the first unit only
            
            observation[agent] = obs_vec

        return observation



class SimpleUnitObservationWrapper_1_1(gym.ObservationWrapper):
    """
    Returns an observation vector for the first unit only
    Included features
    - robot's stats
    - distance vector to closest ice tile
    - distance vector to mother factory (the first factory in the list)
    """
    
    #length of the observation vector for a single robot;
    #13 is the size of the original example code
    #-3 for cargo (all cargo types summed up into a single number instead)
    #-1 for unit type
    #-1 for team id
    SHAPE = 13-3-1-1
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(SimpleUnitObservationWrapper_1_1.SHAPE,)) #one observation vector per robot unit, with added observation of mother factory's water to the units vector

    def observation(self, obs):
        return SimpleUnitObservationWrapper_1_1.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(SimpleUnitObservationWrapper_1_1.SHAPE)

            factories = shared_obs["factories"][agent]
            factory_water = np.zeros(1)  #define variables, in case there is no factory left
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                #get the water stock of the mother factory as an additional observation for this unit, normalized to a reasonable scale
                #factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
                break #returns info about the first factory only
            
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

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
#                        unit["cargo"]["ore"] / cargo_space,
#                        unit["cargo"]["water"] / cargo_space,
#                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
#                unit_type = (
#                    0 if unit["unit_type"] == "LIGHT" else 1
#                )  # note that build actions use 0 to encode Light
                # normalize the unit position

                pos = np.array(unit["pos"]) / env_cfg.map_size

#                unit_vec = np.concatenate([pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1)
                unit_vec = np.concatenate([pos, cargo_vec], axis=-1)

                # we add some engineered features down here
                # compute closest ice tile
                
                ice_tile_distances = np.mean((ice_tile_locations - np.array(unit["pos"])) ** 2, 1)
                closest_ice_tile = (ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size)
                
                obs_vec = np.concatenate([unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1)
                
                break #return info about the first unit only
            
            observation[agent] = obs_vec

        return observation


class SimpleUnitObservationWrapper_1(gym.ObservationWrapper):
    """
    Returns an observation vector for the first unit only
    Included features
    - robot's stats
    - distance vector to closest ice tile
    - distance vector to mother factory (the first factory in the list)
    - water cargo of this factory
    """
    
    SHAPE = 13+1 #length of the observation vector for a single robot; 13 is the size of the original example code
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(SimpleUnitObservationWrapper_1.SHAPE,)) #one observation vector per robot unit, with added observation of mother factory's water to the units vector

    def observation(self, obs):
        return SimpleUnitObservationWrapper_1.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(SimpleUnitObservationWrapper_1.SHAPE)

            factories = shared_obs["factories"][agent]
            factory_water = np.zeros(1)  #define variables, in case there is no factory left
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                
                #get the water stock of the mother factory as an additional observation for this unit, normalized to a reasonable scale
                factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
                break
            
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
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
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos, factory_water], axis=-1   
                )
                break
            observation[agent] = obs_vec

        return observation
