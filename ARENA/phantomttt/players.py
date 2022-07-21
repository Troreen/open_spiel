import numpy as np
import pickle
import os
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp


class DHM:
    """
    Dark hex MCCFR player
    """
    def __init__(self, policy, name):
        self.p_name = name
        self.policy = policy

    def get_action(self, state):
        a_p = self.policy.action_probabilities(state)
        legal_actions = state.legal_actions()
        a_p = [a_p[a] for a in legal_actions]
        return np.random.choice(legal_actions, p=a_p)

class HumanPlayer:
    
    def __init__(self, name):
        self.p_name = name
        
    def get_action(self, state, get_probs=False):
        legal_actions = state.legal_actions()
        print("Legal actions:", legal_actions)
        a = int(input("Enter action: "))
        while a not in legal_actions:
            print("Invalid action... Legal actions:", legal_actions)
            a = int(input("Enter action: "))
        return a, None

class HandCraftedPlayer:
    """
    Hand crafted player.
    """
    def __init__(self, p0_path, p1_path, name):
        self.p_name = name
        self.p0_path = p0_path
        self.p1_path = p1_path
        self.read_data()

    def read_data(self):
        self._policy = [None, None]
        if self.p0_path:
            with open(self.p0_path, "rb") as file:
                self._policy[0] = pickle.load(file)
        if self.p1_path:
            with open(self.p1_path, "rb") as file:
                self._policy[1] = pickle.load(file)

    def get_action(self, state, get_probs=False):
        cur_player = state.current_player()
        if self._policy[cur_player] is None:
            raise ValueError("No policy for player {}".format(cur_player))
        policy_for_state = self._policy[cur_player][state.information_state_string()]
        policy_for_state = {k: v for k, v in policy_for_state}
        action = np.random.choice(list(policy_for_state.keys()), p=list(policy_for_state.values()))
        if get_probs:
            return action, policy_for_state
        return action, None
        
    def action_probabilities(self, state):
        if self._policy[state.current_player()] is None:
            raise ValueError("No policy for player {}".format(state.current_player()))
        return self._policy[state.current_player()]

