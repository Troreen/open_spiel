from player import Player
import numpy as np
import pickle
import os
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp

def_values = {
    'replay_buffer_capacity': int(2e5),
    'reservoir_buffer_capacity': int(2e6),
    'min_buffer_size_to_learn': 1000,
    'anticipatory_param': 0.1,
    'batch_size': 128,
    'learn_every': 64,
    'rl_learning_rate': 0.01,
    'sl_learning_rate': 0.01,
    'optimizer_str': 'sgd',
    'loss_str': 'mse',
    'dropout_rate': 0.2,
    'use_batch_norm': 'True',
    'update_target_network_every': 19200,
    'discount_factor': 1.0,
    'epsilon_decay_duration': int(20e6),
    'epsilon_start': 0.06,
    'epsilon_end': 0.001,
}

class DHN(Player):
    """
    Dark Hex NFSP player.
    """
    
    def __init__(self, num_rows, num_cols, num_actions, 
                 obs_state_size, pone, imperfect_recall,
                 sess, name):
        self.sess = sess
        self.num_rows = num_rows
        self.num_cols = num_cols
        pone_text = "pone" if pone else "npone"
        if imperfect_recall:
            ir_text = "ir"
            self.model_type = "resnet"    
        else:
            ir_text = "pr"
            self.model_type = "mlp"
        self.num_actions = num_actions
        self.obs_state_size = obs_state_size
        self.p_name = name
        self.p_type = "nfsp"
        self.player_info = f"{pone_text}_{ir_text}"
        path = f"tmp/Arena/arena_{self.p_type}_{num_rows}x{num_cols}_{self.player_info}"
        super(DHN, self).__init__(self.p_type, path)

        self.read_data()
        
    def read_data(self):
        self._policies = [
            nfsp.NFSP(
                self.sess,
                idx,
                self.obs_state_size,
                self.num_actions,
                [128],
                model_type=self.model_type,
                input_shape=(3, self.num_rows, self.num_cols),
                **def_values) for idx in range(2)
        ]
        for policy in self._policies:
            policy.restore(self.path)
        self._obs = {
            "info_state": [None] * 2,
            "legal_actions": [None] * 2
        }

    def get_action(self, state):
        a_p = self.action_probabilities(state)
        return np.random.choice(list(a_p.keys()), p=list(a_p.values()))

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(nfsp.MODE.average_policy):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


class DHM:
    """
    Dark hex MCCFR player
    """
    def __init__(self, player_info, policy, name):
        self.player_info = player_info
        self.p_name = name
        self.p_type = "mccfr"
        self.p_name = name
        self.policy = policy

    def get_action(self, state):
        a_p = self.action_probabilities(state)
        return np.random.choice(list(a_p.keys()), p=list(a_p.values()))

    def action_probabilities(self, state, player_id=None):
        a_p = self.policy.action_probabilities(state)
        legal_actions = state.legal_actions()
        a_p = [a_p[a] for a in legal_actions]
        return {a: p for a, p in zip(legal_actions, a_p)}


class HandCraftedPlayer(Player):
    """
    Hand crafted player.
    """
    def __init__(self, num_rows, num_cols, p0_path, p1_path, name):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.p_name = name
        self.player_info = f"pone_ir"
        self.p_type = name
        self.p_name = name
        self.p0_path = p0_path
        self.p1_path = p1_path
        super(HandCraftedPlayer, self).__init__(self.p_type, self.player_info)
        self.read_data()

    def read_data(self):
        self._policy = [None, None]
        if self.p0_path:
            with open(self.p0_path, "rb") as file:
                self._policy[0] = pickle.load(file)
        if self.p1_path:
            with open(self.p1_path, "rb") as file:
                self._policy[1] = pickle.load(file)
        
        if 'strategy' in self._policy[0]:
            self._policy[0] = self._policy[0]['strategy']
        if 'strategy' in self._policy[1]:
            self._policy[1] = self._policy[1]['strategy']

    def get_action(self, state):
        cur_player = state.current_player()
        if self._policy[cur_player] is None:
            raise ValueError("No policy for player {}".format(cur_player))
        policy_for_state = self._policy[cur_player][state.information_state_string()]
        policy_for_state = {k: v for k, v in policy_for_state}
        action = np.random.choice(list(policy_for_state.keys()), p=list(policy_for_state.values()))
        return action
        
    def action_probabilities(self, state):
        if self._policy[state.current_player()] is None:
            raise ValueError("No policy for player {}".format(state.current_player()))
        return self._policy[state.current_player()]
