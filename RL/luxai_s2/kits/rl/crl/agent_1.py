"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""
###Add one feature to observation space, with the water cargo of the mother factory
###this is defined in SimpleUnitObservationWrapper_1
###hopefully this will give a signal to the unit about when to stop digging and return to the factory to deliver water

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

from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper_1

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "models/LuxAI_S2-v0__ppo_train_1__1__1678596265"

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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.agent = AgentModel(obs_dim=SimpleUnitObservationWrapper_1.SHAPE, action_dim=12)
        self.agent.load_state_dict(torch.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH)))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

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
        obs = SimpleUnitObservationWrapper_1.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = torch.from_numpy(obs).float()
        with torch.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                torch.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            
            #features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            #x = self.policy.policy.mlp_extractor.shared_net(features)
            #logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller
            #logits[~action_mask] = -1e8 # mask out invalid actions
            #dist = th.distributions.Categorical(logits=logits)
            #actions = dist.sample().cpu().numpy() # shape (1, 1)
            logits = self.agent.actor(obs.unsqueeze(0)) 
            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)
            
            #https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/policies.py
            #q_values = self.policy.q_net(obs.unsqueeze(0))
            #q_values[~action_mask] = -1.
            #print(q_values, file=sys.stderr)
            #actions = q_values.argmax(dim=1).reshape(-1) # shape (1, N) where N=12 for the default controller
            
            #print(actions, file=sys.stderr)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(self.player, raw_obs, actions[0])

        #watering lichen which can easily improve your agent
        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            if 1000 - step < 50 and factory["cargo"]["water"] > 100:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
