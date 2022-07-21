from players import HandCraftedPlayer, HumanPlayer

import pyspiel
import os
import numpy as np


def play_game(agent, play_as, show_probs=False, show_opp=False):
    """
    Play the game between an agent and a human player.
    """
    game = pyspiel.load_game(f"phantom_ttt_ir")
    state = game.new_initial_state()
    h_player = HumanPlayer("human_player")
    agents = [h_player, agent] if play_as == 0 else [agent, h_player]
    if show_opp:
        print(tree_info_string(state.information_state_string(0), state.information_state_string(1)), '\n')
    else:
        print(tree_info_string_single(state.information_state_string(play_as), play_as), '\n')
    while not state.is_terminal():
        if state.current_player() == 0:
            action, probs = agents[0].get_action(state, get_probs=show_probs)
        else:
            action, probs = agents[1].get_action(state, get_probs=show_probs)
        if show_probs:
            print(probs, '|', action)
        state.apply_action(action)
        if show_opp:
            print(tree_info_string(state.information_state_string(0), state.information_state_string(1)), '\n')
        else:
            print(tree_info_string_single(state.information_state_string(play_as), play_as), '\n')
    return state.returns()

def test_plays():
    agent = HandCraftedPlayer(p0_path="tmp/phantom_ttt_simcap/p0.pkl",
                              p1_path="tmp/phantom_ttt_simcap/p1.pkl",
                              name="simcap_player")

    cont = True
    while cont:
        play_as =       int(input("Play as (0/1):   ")) # 0: black, 1: white
        show_probs =    int(input("Show probs (0/1):")) # 0: no, 1: yes
        show_opps =     int(input("Show opps (0/1): ")) # 0: no, 1: yes
        print("Starting game...\n")
        print("Res:", play_game(agent, play_as, show_probs, show_opps))
        print("\n")
        cont = input("Continue (0/1): ")


def tree_info_string(info_state_0, info_state_1):
    """ Converts the info_state to a string. TIC-TAC-TOE """
    s = ' p0|p1 \n---+---\n'
    is0 = info_state_0[3:].split('\n')
    is1 = info_state_1[3:].split('\n')
    for i, j in zip(is0, is1):
        s += i + '|' + j + '\n'
    return s

def tree_info_string_single(info_state, player):
    """ Converts the info_state to a string. TIC-TAC-TOE """
    s = f' p{player}\n---\n'
    return s + info_state[3:]


if __name__ == "__main__":
    test_plays()
    

    
    
