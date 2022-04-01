import pyspiel
import numpy as np
import tqdm
import time
import pickle

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
  with open(f"tmp/dark_hex_mccfr_4x3/dark_hex_mccfr_solver", "rb") as file:
    solver = pickle.load(file)
  num_games = int(1e7)
  game = pyspiel.load_game("dark_hex_ir(num_rows=4,num_cols=3,use_early_terminal=true)")
  early_ns, early_times = _measure_games(game, num_games, solver.average_policy())
  game = pyspiel.load_game("dark_hex_ir(num_rows=4,num_cols=3,use_early_terminal=false)")
  n_early_ns, n_early_times = _measure_games(game, num_games, solver.average_policy())
  # report
  print("early_ns:", early_ns)
  print("early_times:", early_times)
  print("n_early_ns:", n_early_ns)
  print("n_early_times:", n_early_times)
  

def _measure_games(game, num_games, policy):
  tot_game_length = 0
  start = time.time()
  for _ in tqdm.tqdm(range(num_games)):
    state = game.new_initial_state()
    while not state.is_terminal():
      action_probs = policy.action_probabilities(state)
      action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
      state.apply_action(action)
      tot_game_length += 1
  end = time.time()
  return tot_game_length, (end - start)
      

# test_dark_hex_early_terminal()
# test_dark_hex_num_hidden_stones()
test_counting_states_early_terminal()