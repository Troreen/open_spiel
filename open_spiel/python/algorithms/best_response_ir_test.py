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

"""Tests for open_spiel.python.algorithms.best_response_imperfect_recall."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import dill
import time
import resource

from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
import pyspiel

def report(data, type: str) -> None:
    """ Prints the report in a pretty format given the data
    and the type. Valid types are = ['memory', 'time']

    Time: Output of time.time()
    Memory: Output of process.memory_info().rss
    """
    bold = "\033[1m"
    red = "\033[1;31m"
    yellow = "\033[1;33m"
    green = "\033[1;32m"
    end = "\033[0m"
    if type == 'memory':
        print(f"{bold}{green}Memory usage:\t{end}", end='')
        gbs = data // (1024 ** 2)
        mbs = (data - gbs * 1024 ** 2) // 1024
        kbs = (data - gbs * 1024 ** 2 - mbs * 1024)
        if gbs > 0:
            print(f"{gbs} {red}GB{end} ", end='')
        if mbs > 0:
            print(f"{mbs} {red}MB{end} ", end='')
        if kbs > 0:
            print(f"{kbs} {red}KB{end}", end='')
        print()
    elif type == 'time':
        print(f"{bold}{green}Time taken:\t{end}", end="")
        m, s = divmod(data, 60)
        h, m = divmod(m, 60)
        h, m, s = int(h), int(m), int(s)
        if h > 0:
            print(f"{h}:{m:02d}:{s:02d} {yellow}hours{end}")
        elif m > 0:
            print(f"{m}:{s:02d} {yellow}minutes{end}")
        else:
            print(f"{s:02d} {yellow}seconds{end}")
    else:
        print(f"{red}{bold}Invalid type given to report(). Valid types are = ['memory', 'time']{end}")


def test_best_response_for_partial_ir_policy():
  game = pyspiel.load_game("phantom_ttt_ir")
  test_player = 0
  tabular_policy = {
    # infostate: [(action, prob)]
    "P0 ...\n...\n...": [(4, 1.0)],
    "P0 ...\n.x.\n...": [(0, 1.0)],
    "P0 o..\n.x.\n...": [(1, 1.0)],
    "P0 ox.\n.x.\n...": [(7, 1.0)],
    "P0 ox.\n.x.\n.o.": [(6, 1.0)],
    "P0 ox.\n.x.\nxo.": [(2, 1.0)],
    "P0 oxo\n.x.\nxo.": [(3, 1.0)],
    "P0 oxo\nxx.\nxo.": [(5, 1.0)],
    "P0 oxo\nxxo\nxo.": [(8, 1.0)],
    "P0 x..\n.x.\n...": [(8, 1.0)],
    "P0 x..\n.x.\n..o": [(6, 1.0)],
    "P0 x..\n.x.\no.o": [(7, 1.0)],
    "P0 x..\n.x.\noxo": [(1, 1.0)],
    "P0 xo.\n.x.\noxo": [(3, 1.0)],
    "P0 xo.\nxx.\noxo": [(5, 1.0)],
    "P0 xo.\nxxo\noxo": [(2, 1.0)],
    "P0 x..\n.x.\nx.o": [(2, 1.0)],
    "P0 x.o\n.x.\nx.o": [(3, 1.0)],
    "P0 x.o\nox.\nx.o": [(5, 1.0)],
    "P0 x.o\noxx\nx.o": [(1, 1.0)],
    "P0 xoo\noxx\nx.o": [(7, 1.0)],
  }
  pyspiel_policy = policy.PartialTabularPolicy(game, tabular_policy, test_player)
  br = best_response.BestResponsePolicyIR(game, policy=pyspiel_policy, player_id=1-test_player)
  state_test = game.new_initial_state()
  state_test.apply_action(0)
  br_pi = br.value(state_test)
  print(br_pi)


def test_br_strategy_full_size():
  start = time.time()
  file_path = "tmp/phantom_ttt_p0_simplified.pkl"
  data = dill.load(open(file_path, "rb"))
  game = pyspiel.load_game("phantom_ttt_ir")
  player_id = data["player"]
  pyspiel_policy = policy.PartialTabularPolicy(
      game, policy=data["strategy"], player=player_id)
  br = best_response.BestResponsePolicyIR(
      game, policy=pyspiel_policy, player_id=1-player_id)
  state_test = game.new_initial_state()
  # state_test.apply_action(0)
  br_pi = br.value(state_test)
  print(br_pi)
  report(time.time() - start, 'time')
  report(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'memory')


# test_best_response_for_partial_ir_policy()
test_br_strategy_full_size()
