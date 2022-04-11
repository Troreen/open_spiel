from players import DHN, DHM, HandCraftedPlayer

import pyspiel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow.compat.v1 as tf


def play_game(agent0, agent1, num_rows, num_cols):
    """
    Play the game between two agents.
    """
    game_ir_npone = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols})")
    game_ir_pone = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols},use_early_terminals=True)")
    game_pr = pyspiel.load_game(f"dark_hex(num_rows={num_rows},num_cols={num_cols})")
    
    state_ir_npone = game_ir_npone.new_initial_state()
    state_ir_pone = game_ir_pone.new_initial_state()
    state_pr = game_pr.new_initial_state()

    pone_over = False

    while not state_pr.is_terminal():
        if not pone_over:
            if agent0.player_info == "pone_ir" and state_ir_pone.is_early_terminal()[1] == 0:
                return [-1, 1]
            if agent1.player_info == "pone_ir" and state_ir_pone.is_early_terminal()[1] == 1:
                return [1, -1]
        if state.current_player() == 0:
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
        state_ir_npone.apply_action(action)
        state_pr.apply_action(action)
        if not pone_over:
            state_ir_pone.apply_action(action)
            if state_ir_pone.is_terminal():
                pone_over = True
    return state_pr.returns()


def play_games(agent_a, agent_b, n):
    """
    Play n games between two agents. Agents play n/2 games as
    player 0 and n/2 games as player 1.
    """
    wins = [0, 0]
    for _ in range(n // 2):
        res = play_game(agent_a, agent_b)
        wins[0] += res[0]
        wins[1] += res[1]
    for _ in range(n // 2):
        res = play_game(agent_b, agent_a)
        wins[0] += res[1]
        wins[1] += res[0]
    return wins


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
            DHM(num_rows=num_rows, num_cols=num_cols, pone=False, imperfect_recall=False),
            # MCCFR - Imperfect Recall with no pONE
            DHM(num_rows=num_rows, num_cols=num_cols, pone=False, imperfect_recall=True),
            # MCCFR - Imperfect Recall with pONE
            DHM(num_rows=num_rows, num_cols=num_cols, pone=True, imperfect_recall=True),
            # Handcrafted - Ryan's Player
            HandCraftedPlayer(num_rows=num_rows, num_cols=num_cols,
                              p0_path="tmp/ryan_player/p0_strategy.pkl",
                              p1_path="tmp/ryan_player/p1_strategy.pkl")
        ]

        print("\033[1m\033[33m" + "Starting games..." + "\033[0m")

        records = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                p0_name = f"{agents[i].p_type}_{agents[i].player_info}"
                p1_name = f"{agents[j].p_type}_{agents[j].player_info}"
                print(f"\033[1m\033[32m" + f"Playing {p0_name} vs {p1_name}..." + "\033[0m")
                wins = play_games(agents[i], agents[j], n)
                print(f"{p0_name} vs {p1_name}: {wins[0]} - {wins[1]}")
                records.append([p0_name, p1_name, wins[0], wins[1]])
    return records


if __name__ == "__main__":
    arena(5)