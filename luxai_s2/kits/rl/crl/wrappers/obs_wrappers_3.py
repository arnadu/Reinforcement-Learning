from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces



    

SHAPE = 13+1 #length of the observation vector for a single robot

class SimpleUnitObservationWrapper_3(gym.ObservationWrapper):
    """
    Included features: for each unit
    - robot's stats
    - distance vector to closest ice tile
    - distance vector to mother factory
    - water cargo of this factory
    """

    
    def __init__(self, env: gym.Env, control_tower) -> None:
        super().__init__(env)
        #self.observation_space = spaces.Box(-999, 999, shape=(SHAPE,)) #one observation vector per robot unit, with added observation of mother factory's water to the units vector
        self.observation_space = spaces.Space()
        self.control_tower = control_tower  #used to add data to the observation

    def observation(self, obs):
        return SimpleUnitObservationWrapper_3.obs_for_units(obs, self.env.state.env_cfg, self.control_tower)

    #wrapper to prepare a list of observation vectors, one for each heavy unit, that can be used by the NN to find the best action for each unit
    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def obs_for_units(obs, env_cfg, control_tower):

        shared_obs = obs['player_0']
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        observations = {} #return a list of observations that can be passed to the NN, one for each unit

        for player in ['player_0', 'player_1']:

            factories = shared_obs['factories'][player]
            factories_locations = np.array([ np.array(factories[factory_id]['pos']) for factory_id in factories.keys()])
            
            units = shared_obs["units"][player]
            if len(units)>0:
                
                observations[player]=np.zeros((len(units), SHAPE))  #one observation vector per robot unit
                for i, (k, unit) in enumerate(units.items()):
        
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
                    closest_ice_tile = (
                        ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                    )
        
                    '''
                    '''
                    #tell the unit to focus on its mother factory
                    #the control tower maintains a mapping of each unit to its mother factory
                    if (k not in control_tower[player]['units_factories']) \
                        or (control_tower[player]['units_factories'][k] not in factories):  
                        #the unit has just been created and is not yet associated with a factory
                        #or the mother factory has died and we need to focus on a different one
                        
                        # identify the closest factory
                        factories_distances = np.mean(
                            (factories_locations - np.array(unit["pos"])) ** 2, 1
                        )
                        i = np.argmin(factories_distances)
                        factory_id = list(factories)[i]  #rely on the fact that dictionaries are order-preserving
                        control_tower[player]['units_factories'][k]=factory_id
                    
                    #we want the unit to focus on this specific factory and not wander off
                    factory_id = control_tower[player]['units_factories'][k]
                    factory = factories[factory_id] 
                    factory_vec = factory['pos']/env_cfg.map_size
                    factory_water = np.array([float(factory['cargo']['water']) / 1000.])               
                    
                    #put the entire observation together
                    obs_vec = np.concatenate(
                        [unit_vec, factory_vec - pos, closest_ice_tile - pos, factory_water], axis=-1
                    )
                    
                    observations[player][i] = obs_vec
            else:
                observations[player] = np.zeros(SHAPE)
                
        return observations




    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                13+1,
            )

            factories = shared_obs["factories"][agent]
            factory_water = np.zeros(1)  #define variables, in case there is no factory left
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                
                #get the water stock of the mother factory as an additional observation for this unit
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
