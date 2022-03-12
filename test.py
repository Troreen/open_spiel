import pyspiel
import numpy as np
import tqdm

def test_dark_hex_early_terminal():
  game = pyspiel.load_game(
    "dark_hex_ir(board_size=2,use_early_terminal=true)")
  state = game.new_initial_state()
  state.apply_action(0)
  is_terminal = state.is_terminal()
  is_early_terminal = state.is_early_terminal()
  assert is_terminal == False
  assert is_early_terminal == (False, -3)
  state.apply_action(1)
  is_terminal = state.is_terminal()
  is_early_terminal = state.is_early_terminal()
  assert is_terminal == True
  assert is_early_terminal == (True, 0)
  print("test_dark_hex_early_terminal passed")

def test_dark_hex_num_hidden_stones():
  game = pyspiel.load_game("dark_hex_ir(board_size=2)")
  state = game.new_initial_state()
  state.apply_action(0)
  state.apply_action(1)
  assert state.num_hidden_stones(0) == 1
  assert state.num_hidden_stones(1) == 1
  state.apply_action(1)
  assert state.num_hidden_stones(0) == 0
  print("test_dark_hex_num_hidden_stones passed")

def test_counting_states_early_terminal():
  num_games = 1000
  game = pyspiel.load_game("dark_hex_ir(num_rows=4,num_cols=3,use_early_terminal=true)")
  num_states_early_term = _measure_games(game, num_games)
  game = pyspiel.load_game("dark_hex_ir(num_rows=4,num_cols=3)")
  num_states_no_early_term = _measure_games(game, num_games)
  # assert num_states_early_term < num_states_no_early_term
  print(f"Number of states with early terminal: {num_states_early_term}")
  print(f"Number of states without early terminal: {num_states_no_early_term}")

def _measure_games(game, num_games):
  tot_game_length = 0
  for _ in tqdm.tqdm(range(num_games)):
    state = game.new_initial_state()
    while not state.is_terminal():
      rand_action = state.legal_actions()[np.random.randint(len(state.legal_actions()))]
      state.apply_action(rand_action)
      tot_game_length += 1
    print(state.returns())
  return tot_game_length / num_games
      

test_dark_hex_early_terminal()
test_dark_hex_num_hidden_stones()
test_counting_states_early_terminal()