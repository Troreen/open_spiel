from players import DHN, DHM, HandCraftedPlayer

import pyspiel
import os
from tqdm import tqdm
import pickle
import csv
import numpy as np
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow.compat.v1 as tf


def play_game(agent0, agent1, num_rows, num_cols):
    """
    Play the game between two agents.
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
                return [1, -1]
            if agent1.player_info == "pone_ir" and state_ir_pone.is_early_terminal()[1] == 1:
                # print("pone_ir wins with pone")
                return [-1, 1]
        if state_pr.current_player() == 0:
            if agent0.player_info == "npone_pr":
                action = agent0.get_action(state_pr)
            elif agent0.player_info == "pone_ir" and not pone_over:
                action = agent0.get_action(state_ir_pone)
            else:
                action = agent0.get_action(state_ir_npone)
            # print(f"p0 action: {action}")
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
    return state_pr.returns()


def play_games(agent_a, agent_b, n, num_rows, num_cols):
    """
    Play n games between two agents. Agents play n/2 games as
    player 0 and n/2 games as player 1.
    """
    p0_results = [0, 0]
    p1_results = [0, 0]
    for _ in tqdm(range(n // 2), desc=f"Black: {agent_a.player_info} vs White: {agent_b.player_info}"):
        res = play_game(agent_a, agent_b, num_rows, num_cols)
        p0_results[0] += res[0]
        p1_results[1] += res[1]
    for _ in tqdm(range(n // 2), desc=f"Black: {agent_b.player_info} vs White: {agent_a.player_info}"):
        res = play_game(agent_b, agent_a, num_rows, num_cols)
        p1_results[0] += res[0]
        p0_results[1] += res[1]
    return p0_results, p1_results


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
            # NFSP - Perfect Recall
            DHN(num_rows=num_rows, num_cols=num_cols, num_actions=pr_action_size,
                obs_state_size=pr_obs_size, pone=False, imperfect_recall=False,
                sess=sess),
            # NFSP - Imperfect Recall with no pONE
            DHN(num_rows=num_rows, num_cols=num_cols, num_actions=ir_action_size,
                obs_state_size=ir_obs_size, pone=False, imperfect_recall=True,
                sess=sess),
            # NFSP - Imperfect Recall with pONE
            DHN(num_rows=num_rows, num_cols=num_cols, num_actions=ir_action_size,
                obs_state_size=ir_obs_size, pone=True, imperfect_recall=True,
                sess=sess),
            # MCCFR - Perfect Recall
            # DHM('npone_pr', npone_pr_policy),
            # MCCFR - Imperfect Recall with no pONE
            DHM('pone_ir', pone_ir_policy),
            # MCCFR - Imperfect Recall with pONE
            DHM('npone_ir', npone_ir_policy),
            # Handcrafted - Ryan's Player
            HandCraftedPlayer(num_rows=num_rows, num_cols=num_cols,
                              p0_path="tmp/ryan_player/p0_strategy.pkl",
                              p1_path="tmp/ryan_player/p1_strategy.pkl")
        ]

        print("\033[1m\033[33m" + "Starting games..." + "\033[0m")

        records = defaultdict(lambda: {'p0': 0, 'p1': 0})   
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                p0_name = f"{agents[i].p_type}_{agents[i].player_info}"
                p1_name = f"{agents[j].p_type}_{agents[j].player_info}"
                print(f"\033[1m\033[32m" + f"Playing {p0_name} vs {p1_name}..." + "\033[0m")
                results_p0, results_p1 = play_games(agents[i], agents[j], n, num_rows, num_cols)
                records[p0_name]['p0'] += results_p0[0]
                records[p0_name]['p1'] += results_p0[1]
                records[p1_name]['p0'] += results_p1[0]
                records[p1_name]['p1'] += results_p1[1]
                print(f"\033[1m\033[32m" + f"{p0_name} vs {p1_name} finished." + "\033[0m")
    print(records)
    return records


if __name__ == "__main__":
    records = arena(200)

    # Make 3 different rankings:
    # 1. Wins as p0
    # 2. Wins as p1
    # 3. Wins as p0 and p1
    # Sort the players for each ranking and save the results to a csv file.

    # Sort the players by wins as p0 and p1
    p0_sorted = sorted(records.items(), key=lambda x: x[1]['p0'], reverse=True)
    p1_sorted = sorted(records.items(), key=lambda x: x[1]['p1'], reverse=True)
    p0_and_p1_sorted = sorted(records.items(), key=lambda x: x[1]['p0'] + x[1]['p1'], reverse=True)

    # Save the results to a csv file
    if not os.path.exists("tmp/Arena/res/"):
        os.makedirs("tmp/Arena/res/")
    # p0_sorted
    with open("tmp/Arena/res/p0_sorted.csv", "w") as file:
        file.write("Player,Wins\n")
        for player in p0_sorted:
            file.write(f"{player[0]},{player[1]['p0']}\n")
    # p1_sorted
    with open("tmp/Arena/res/p1_sorted.csv", "w") as file:
        file.write("Player,Wins\n")
        for player in p1_sorted:
            file.write(f"{player[0]},{player[1]['p1']}\n")
    # p0_and_p1_sorted
    with open("tmp/Arena/res/p0_and_p1_sorted.csv", "w") as file:
        file.write("Player,Wins\n")
        for player in p0_and_p1_sorted:
            file.write(f"{player[0]},{player[1]['p0'] + player[1]['p1']}\n")

    # Print the results
    print("\033[1m\033[32m" + "Results:" + "\033[0m")
    print("\033[1m\033[32m" + "p0_sorted:" + "\033[0m")
    for player in p0_sorted:
        print(f"{player[0]}: {player[1]['p0']}")
    print("\033[1m\033[32m" + "p1_sorted:" + "\033[0m")
    for player in p1_sorted:
        print(f"{player[0]}: {player[1]['p1']}")
    print("\033[1m\033[32m" + "p0_and_p1_sorted:" + "\033[0m")
    for player in p0_and_p1_sorted:
        print(f"{player[0]}: {player[1]['p0'] + player[1]['p1']}")
