# load strategy from file
# examine by playing against it
#
#
import pyspiel
import pickle
import numpy as np


def play_against_strategy(strategy, game, player=0):
    """Play against a strategy"""
    state = game.new_initial_state()
    while not state.is_terminal():
        print(state.information_state_string(player))
        print(state.information_state_string(1 - player))
        if state.current_player() == player:
            legal_actions = state.legal_actions()
            action_probs = strategy.action_probabilities(state)
            print(f"Action probs: {action_probs}")
            print(f"Legal actions: {legal_actions}")
            action = input("Enter action: ")
            state.apply_action(int(action))
        else:
            action_probs = strategy.action_probabilities(state)
            print(f"Action probs: {action_probs}")
            action = -1 #int(input("Enter action (opp): "))
            if action == -1:
                action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
            state.apply_action(action)
    print(f"Final info state p: {state.information_state_string(player)}")
    print(f"Final info state o: {state.information_state_string(1 - player)}")
    print(f"Final score: {state.returns()}")


def main(num_rows, num_cols):
    # load game
    game = pyspiel.load_game(f"dark_hex_ir(num_rows={num_rows},num_cols={num_cols},use_early_terminal=True)")
    # load solver
    with open(f"tmp/dark_hex_mccfr_{num_rows}x{num_cols}/dark_hex_mccfr_solver", "rb") as file:
        solver = pickle.load(file)
    # play against strategy
    play_against_strategy(solver.average_policy(), game)


if __name__ == "__main__":
    main(4, 3)
        

    