from player import Player
import numpy as np
import pickle

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
                 sess):
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

        self.p_type = "nfsp"
        self.player_info = f"{pone_text}_{ir_text}"
        path = f"tmp/arena_{p_type}_{num_rows}x{num_cols}_{self.player_info}"
        super(DHN, self).__init__(self.p_type, path)

        self.read_data()
        
    def read_data(self):
        self._policies = [
            nfsp.NFSP(
                self.sess,
                idx,
                self.obs_state_size,
                self.num_actions,
                [126],
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

class DHM(Player):
    """
    Dark hex MCCFR player
    """
    def __init__(self, num_rows, num_cols, num_actions, 
                 obs_state_size, pone, imperfect_recall):
        self.num_rows = num_rows
        self.num_cols = num_cols
        pone_text = "pone" if pone else "npone"
        ir = "ir" if imperfect_recall else "pr"
        self.player_info = f"{pone_text}_{ir}"
        self.p_type = "mccfr"
        f"tmp/arena_{self.p_type}_{num_rows}x{num_cols}_{self.player_info}/dark_hex_mccfr_solver"

    def read_data(self):
        with open(self.path, "rb") as file:
            solver = pickle.load(file)
        self._policy = solver.average_policy()

    def get_action(self, state):
        return np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))

    def action_probabilities(self, state, player_id=None):
        return self._policy.action_probabilities(state)


class HandCraftedPlayer(Player):
    """
    Hand crafted player.
    """
    def __init__(self, num_rows, num_cols, p0_path, p1_path):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.player_info = f"pone_ir"
        self.p_type = "handcrafted"

    def read_data(self):
        self._policy = [None, None]
        if self.p0_path:
            with open(self.p0_path, "rb") as file:
                self._policy[0] = pickle.load(file)
            actions = [x for x, _ in self._policy[0]]
            probs = [x for _, x in self._policy[0]]
            self._policy[0] = dict(zip(actions, probs))
        if self.p1_path:
            with open(self.p1_path, "rb") as file:
                self._policy[1] = pickle.load(file)
            actions = [x for x, _ in self._policy[1]]
            probs = [x for _, x in self._policy[1]]
            self._policy[1] = dict(zip(actions, probs))

    def get_action(self, state):
        cur_player = state.current_player()
        if self._policy[cur_player] is None:
            raise ValueError("No policy for player {}".format(cur_player))
        action = np.random.choice(list(self._policy[cur_player].keys()), p=list(self._policy[cur_player].values()))
        return action
        
    def action_probabilities(self, state):
        if self._policy[state.current_player()] is None:
            raise ValueError("No policy for player {}".format(state.current_player()))
        return self._policy[state.current_player()]
