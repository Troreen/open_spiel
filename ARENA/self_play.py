from players import HandCraftedPlayer

import pyspiel
from tqdm import tqdm
import pickle
import numpy as np

games_lost = 0

def play_game(agent, num_rows, num_cols):
    """
    Play the game between two agents.
    """
    game_ir_pone = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols},use_early_terminal=True)")
    game_ir = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols})")
    
    state_ir_pone = game_ir_pone.new_initial_state()
    state_ir = game_ir.new_initial_state()

    pone_over = False

    while not state_ir.is_terminal():
        if not pone_over and state_ir_pone.is_early_terminal()[0]:
            pone_over = True
        action = agent.get_action(state_ir)
        state_ir.apply_action(action)
        if not pone_over:
            state_ir_pone.apply_action(action)
    return state_ir.returns(), state_ir_pone.returns()


def player_test(agent, num_rows, num_cols):
    """
    Play n games between two agents. Agents play n/2 games as
    player 0 and n/2 games as player 1.
    """
    global games_lost
    results = [0, 0]
    for _ in tqdm(range(int(3e5))):
        ir_res, pone_res = play_game(agent, num_rows, num_cols)
        if ir_res[0] != pone_res[0]:
            games_lost += 1
        results[0] += ir_res[0]
        results[1] += ir_res[1]
    print(f"games lost: {games_lost}")
    print(f"res: {results}")


if __name__ == "__main__":
    num_rows = 4
    num_cols = 3
    p = HandCraftedPlayer(num_rows=num_rows, num_cols=num_cols,
                        p0_path="tmp/simplified_mccfr/p0_strategy.pkl",
                        p1_path="tmp/simplified_mccfr/p1_strategy.pkl",
                        name="Simplified MCCFR")
    player_test(p, num_rows, num_cols)
