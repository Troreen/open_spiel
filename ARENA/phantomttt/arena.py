from players import DHM, HandCraftedPlayer

import pyspiel
import os
from tqdm import tqdm
import pickle
import csv
import numpy as np
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow.compat.v1 as tf


def play_game(agent0, agent1):
    """
    Play the game between two agents.
    """
    game = pyspiel.load_game(f"phantom_ttt_ir")
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.current_player() == 0:
            action = agent0.get_action(state)
        else:
            action = agent1.get_action(state)
        state.apply_action(action)
    return state.returns()


def play_games(agent_a, agent_b, n):
    """
    Play n games between two agents. Agents play n/2 games as
    player 0 and n/2 games as player 1.
    """
    p0_results = [0, 0]
    p1_results = [0, 0]
    for _ in tqdm(range(n // 2), desc=f"Black: {agent_a.p_name} vs White: {agent_b.p_name}"):
        res = play_game(agent_a, agent_b)
        p0_results[0] += res[0]
        p1_results[1] += res[1]
    for _ in tqdm(range(n // 2), desc=f"Black: {agent_b.p_name} vs White: {agent_a.p_name}"):
        res = play_game(agent_b, agent_a)
        p1_results[0] += res[0]
        p0_results[1] += res[1]
    return p0_results, p1_results


def arena(n):
    """
    Setup all the players and play n games in round robin.
    """
    print("\033[1m" + "Arena" + "\033[0m")
    print("\033[1m\033[32m" + "Setting up agents..." + "\033[0m")

    _ir_game = pyspiel.load_game(f"phantom_ttt_ir")
    ir_obs_size = _ir_game.information_state_tensor_size()
    ir_action_size = _ir_game.num_distinct_actions()

    with open("tmp/phantom_ttt_ir/solver.pkl", "rb") as file:
        solver = pickle.load(file)
    p_policy = solver.average_policy()

    agents = [
        # MCCFR - Imperfect Recall with pONE
        DHM(p_policy, 'mccfr_player'),
        # SIMCAP
        HandCraftedPlayer(p0_path="tmp/phantom_ttt_simcap/p0.pkl",
                            p1_path="tmp/phantom_ttt_simcap/p1.pkl",
                            name="simcap_player"),
    ]

    print("\033[1m\033[33m" + "Starting games..." + "\033[0m")

    records = defaultdict(lambda: {'p0': 0, 'p1': 0})   
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            p0_name, p1_name = f"{agents[i].p_name}", f"{agents[j].p_name}"
            print(f"\033[1m\033[32m" + f"Playing {p0_name} vs {p1_name}..." + "\033[0m")
            results_p0, results_p1 = play_games(agents[i], agents[j], n)
            records[p0_name]['p0'] += results_p0[0]
            records[p0_name]['p1'] += results_p0[1]
            records[p1_name]['p0'] += results_p1[0]
            records[p1_name]['p1'] += results_p1[1]
            print(f"\033[1m\033[32m" + f"{p0_name} vs {p1_name} finished." + "\033[0m")
    print(records)
    return records


if __name__ == "__main__":
    records = arena(50000)
    print(records)
    

    
    
