from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces



class SimpleUnitObservationWrapper_1(gym.ObservationWrapper):
    """
    Returns an observation vector for the first unit only
    Included features
    - robot's stats
    - distance vector to closest ice tile
    - distance vector to mother factory (the first factory in the list)
    - water cargo of this factory
    """
    
    #length of the observation vector for a single robot;
    #13 is the size of the original example code
    #+1 for the water cargo of the unit's mother factory
    #+8 for the content of the cells adjacent to this unit
    SHAPE = 13+1+8  
    
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
                
                neighbours = np.zeros(8) #8 adjacent cells
                
                
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos, factory_water, neighbours], axis=-1   
                )
                break
            observation[agent] = obs_vec

        return observation
