from players import DHN, DHM, HandCraftedPlayer

import pyspiel
import os
from tqdm import tqdm
import pickle
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow.compat.v1 as tf
from heat_map import heat_map_driver


def play_game(agent0, agent1, num_rows, num_cols):
    """
    Play the game between two agents.

    Returns the first players results.
    """
    game_ir_npone = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols})")
    game_ir_pone = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols},use_early_terminal=True)")
    game_pr = pyspiel.load_game(f"dark_hex(num_rows={num_rows},num_cols={num_cols})")
    
    state_ir_npone = game_ir_npone.new_initial_state()
    state_ir_pone = game_ir_pone.new_initial_state()
    state_pr = game_pr.new_initial_state()

    pone_over = False

    while not state_pr.is_terminal():
        # print(state_ir_pone.information_state_string())
        if not pone_over:
            if state_ir_pone.is_early_terminal()[0]:
                pone_over = True
            if agent0.player_info == "pone_ir" and state_ir_pone.is_early_terminal()[1] == 0:
                # print("pone_ir wins with early terminal")
                return 1
            if agent1.player_info == "pone_ir" and state_ir_pone.is_early_terminal()[1] == 1:
                # print("pone_ir wins with pone")
                return -1
        if state_pr.current_player() == 0:
            if agent0.player_info == "npone_pr":
                action = agent0.get_action(state_pr)
            elif agent0.player_info == "pone_ir" and not pone_over:
                action = agent0.get_action(state_ir_pone)
            else:
                action = agent0.get_action(state_ir_npone)
        else:
            if agent1.player_info == "npone_pr":
                action = agent1.get_action(state_pr)
            elif agent1.player_info == "pone_ir" and not pone_over:
                action = agent1.get_action(state_ir_pone)
            else:
                action = agent1.get_action(state_ir_npone)
            # print(f"p1 action: {action}")
        state_ir_npone.apply_action(action)
        state_pr.apply_action(action)
        if not pone_over:
            state_ir_pone.apply_action(action)
    return int(state_pr.returns()[0])


def play_games(agent_a, agent_b, n, num_rows, num_cols):
    """
    Play n games between two agents. Agent_a is the first player.

    Returns the first players results.
    """
    p0_results = 0
    for _ in tqdm(range(n), desc=f"Black: {agent_a.player_info} vs White: {agent_b.player_info}"):
        p0_results += play_game(agent_a, agent_b, num_rows, num_cols)
    return p0_results


def arena(n):
    """
    Setup all the players and play n games in round robin.
    """
    num_rows = 4
    num_cols = 3
    print("\033[1m" + "Arena" + "\033[0m")
    print("\033[1m\033[32m" + "Setting up agents..." + "\033[0m")

    _pr_game = pyspiel.load_game(f"dark_hex(num_rows={num_rows},num_cols={num_cols})")
    pr_obs_size = _pr_game.information_state_tensor_size()
    pr_action_size = _pr_game.num_distinct_actions()
    _ir_game = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols})")
    ir_obs_size = _ir_game.information_state_tensor_size()
    ir_action_size = _ir_game.num_distinct_actions()

    # with open("tmp/Arena/arena_mccfr_4x3_npone_pr/solver.pkl", "rb") as file:
    #     solver = pickle.load(file)
    # npone_pr_policy = solver.average_policy()
    with open("tmp/Arena/arena_mccfr_4x3_pone_ir/solver.pkl", "rb") as file:
        solver2 = pickle.load(file)
    pone_ir_policy = solver2.average_policy()
    with open("tmp/Arena/arena_mccfr_4x3_npone_ir/solver.pkl", "rb") as file:
        solver3 = pickle.load(file)
    npone_ir_policy = solver3.average_policy()

    with tf.Session() as sess:
        agents = [
            # Simplified MCCFR - SIMCAP+
            HandCraftedPlayer(num_rows=num_rows, num_cols=num_cols,
                              p0_path="tmp/Arena/simcap+/p0_strategy.pkl",
                              p1_path="tmp/Arena/simcap+/p1_strategy.pkl",
                              name="SIMCAP+"),
            # NFSP - Perfect Recall
            DHN(num_rows=num_rows, num_cols=num_cols, num_actions=pr_action_size,
                obs_state_size=pr_obs_size, pone=False, imperfect_recall=False,
                sess=sess, name="NFSP-PR"),
            # NFSP - Imperfect Recall with no pONE
            DHN(num_rows=num_rows, num_cols=num_cols, num_actions=ir_action_size,
                obs_state_size=ir_obs_size, pone=False, imperfect_recall=True,
                sess=sess, name="NFSP-IR"),
            # NFSP - Imperfect Recall with pONE
            DHN(num_rows=num_rows, num_cols=num_cols, num_actions=ir_action_size,
                obs_state_size=ir_obs_size, pone=True, imperfect_recall=True,
                sess=sess, name="NFSP-IR-p"),
            # MCCFR - Perfect Recall
            DHM('npone_pr', npone_pr_policy, name="MCCFR-PR"),
            # MCCFR - Imperfect Recall with no pONE
            DHM('pone_ir', pone_ir_policy, name="MCCFR-IR"),
            # MCCFR - Imperfect Recall with pONE
            DHM('npone_ir', npone_ir_policy, name="MCCFR-IR-p"),
            # Handcrafted - Ryan's Player
            HandCraftedPlayer(num_rows=num_rows, num_cols=num_cols,
                              p0_path="tmp/Arena/ryan_player/p0_strategy.pkl",
                              p1_path="tmp/Arena/ryan_player/p1_strategy.pkl",
                              name="HP"),
            # Simplified MCCFR - SIMCAP
            HandCraftedPlayer(num_rows=num_rows, num_cols=num_cols,
                              p0_path="tmp/Arena/simcap/p0_strategy.pkl",
                              p1_path="tmp/Arena/simcap/p1_strategy.pkl",
                              name="SIMCAP"),
        ]

        print("\033[1m\033[33m" + "Starting games..." + "\033[0m")

        # save the games in n x n table (dict - n is the number of players)
        # each row is a player (as first player) and each column is a player (as second player)
        records = {}
        for i in range(len(agents)):
            p0_name = f"{agents[i].p_name}"
            records[p0_name] = {}
            for j in range(len(agents)):
                if i == j:
                    continue
                p1_name = f"{agents[j].p_name}"
                print(f"\033[1m\033[32m" + f"Playing {p0_name} vs {p1_name}..." + "\033[0m")
                res = play_games(agents[i], agents[j], n, num_rows, num_cols)
                records[p0_name][p1_name] = res
                print(f"\033[1m\033[32m" + f"{p0_name} vs {p1_name} finished." + "\033[0m")
    return records


if __name__ == "__main__":
    records = arena(5000)
    with open("tmp/Arena/res/records.pkl", "wb") as file:
        pickle.dump(records, file)

    # with open("tmp/Arena/res/records.pkl", "rb") as file:
    #     records = pickle.load(file)

    heat_map_driver(
        records=records,
        title="Arena (Average Reward)",
        show_ratio=True,
        num_games=5000,
        save_to_path="tmp/Arena/res/main_arena/"
    )
