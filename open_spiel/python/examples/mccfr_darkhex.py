# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the C++ MCCFR algorithms on Kuhn Poker.

This examples calls the underlying C++ implementations via the Python bindings.
Note that there are some pure Python implementations of some of these algorithms
in python/algorithms as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
from absl import flags

import pyspiel
from tqdm import tqdm
import numpy as np
import os

from functools import partial
from itertools import repeat
import multiprocessing as mp
from contextlib import contextmanager

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", int(5e6), "Number of iterations")
flags.DEFINE_string("game", "dark_hex_ir", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("eval_freq", int(5e4), "How often to run evaluation")
flags.DEFINE_integer("num_eval_games", int(1e4), "Number of games to evaluate")


def main(_):
  num_rows = 4
  num_cols = 3
  game = pyspiel.load_game(
      FLAGS.game,
      {"num_rows": num_rows,
       "num_cols": num_cols,
       "use_early_terminal": True},
  )
  folder_path = f"tmp/dark_hex_mccfr_{num_rows}x{num_cols}"
  # create folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  solver = pyspiel.OutcomeSamplingMCCFRSolver(game)
  rand_res = []
  for i in tqdm(range(FLAGS.iterations)):
    solver.run_iteration()
    if i % FLAGS.eval_freq == 0:
      policy = solver.average_policy()
      rand_eval = run_random_games(game, policy, FLAGS.num_eval_games)
      print(f"Ep {i}; Rand eval: {rand_eval}")

      print("Persisting the model...")
      with open(f"{folder_path}/dark_hex_mccfr_solver", "wb") as file:
        pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

      rand_res.append(rand_eval)
      with open(f"{folder_path}/rand_res.pkl", "wb") as file:
        pickle.dump(rand_res, file, pickle.HIGHEST_PROTOCOL)


def run_random_games(game, policy, num_games):
  """Runs random games and returns average score."""
  scores_as_p = [0., 0.]
  games_per_p = num_games // 2
  
  for _ in range(games_per_p):
    scores_as_p[0] += run_random_game(game, policy, 0) / games_per_p
    scores_as_p[1] += run_random_game(game, policy, 1) / games_per_p
  return scores_as_p


def run_random_game(game, policy, player):
  """Runs a single random game and returns score as player."""
  state = game.new_initial_state()
  while not state.is_terminal():
    legal_actions = state.legal_actions()
    if state.current_player() == player:
      action_probs = policy.action_probabilities(state)
      action_probs = [action_probs[a] for a in legal_actions]
    else:
      action_probs = [1. / len(legal_actions)] * len(legal_actions)
    action = np.random.choice(legal_actions, p=action_probs)
    state.apply_action(action)
  return state.returns()[player]


if __name__ == "__main__":
  app.run(main)
