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
"""NFSP agents trained on Dark Hex."""

from absl import app
from absl import flags
from absl import logging

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp

import json
import numpy as np
from proglog import default_bar_logger
import pickle
import pyspiel

logger = default_bar_logger('bar')

# Log colors
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "dark_hex_ir", "Name of the game.")
flags.DEFINE_integer("num_rows", 4, "Number of rows.")
flags.DEFINE_integer("num_cols", 3, "Number of cols.")
flags.DEFINE_integer("num_players", 2, "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(1e7),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", int(1e5),
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("num_eval_games", int(1e4),
                     "Number of evaluation games when running random_games evaluator.")
flags.DEFINE_list("hidden_layers_sizes", [128], 
                     "Number of hidden units to use in each layer of the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_string("model_type", "resnet", 
                   "Model type, choose from 'resnet' and 'conv2d'.") 
flags.DEFINE_float("dropout_rate", 0.2, "Dropout rate.")
flags.DEFINE_string("use_batch_norm", "True", "Whether to use batch norm.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")
flags.DEFINE_string("evaluation_metric", "random_games",
                    "Choose from 'exploitability', 'nash_conv', 'random_games'.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")


class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode, num_players):
    game = env.game
    player_ids = list(range(num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * num_players,
        "legal_actions": [None] * num_players
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(unused_argv):
  logger(message=f"{OKBLUE}Loading game {FLAGS.game_name}{ENDC}")
  game = FLAGS.game_name
  num_players = FLAGS.num_players
  pone = False
  
  env_configs = {"num_rows": FLAGS.num_rows, "num_cols": FLAGS.num_cols,
                 "use_early_terminal": pone}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": FLAGS.anticipatory_param,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
      "use_batch_norm": FLAGS.use_batch_norm,
      "dropout_rate": FLAGS.dropout_rate,
  }
  pone_text = "pone" if pone else "npone"
  ir = "ir" if FLAGS.game_name == "dark_hex_ir" else "pr"

  checkpoint_dir = f"arena_nfsp_{FLAGS.num_rows}x{FLAGS.num_cols}_{pone_text}_{ir}"

  if FLAGS.use_checkpoints:
    # Create the folder if it doesn't exist.
    if not tf.gfile.Exists(checkpoint_dir):
      tf.gfile.MakeDirs(checkpoint_dir)

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(
            sess,
            idx,
            info_state_size,
            num_actions,
            hidden_layers_sizes,
            # model_type='mlp',
            model_type=FLAGS.model_type,
            input_shape=(3, FLAGS.num_rows, FLAGS.num_cols),
            **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy, FLAGS.num_players)

    sess.run(tf.global_variables_initializer())

    game_res = []
    nash_res = []
    if FLAGS.use_checkpoints:
      for agent in agents:
        if agent.has_checkpoint(checkpoint_dir):
          agent.restore(checkpoint_dir)
      # load the random game results if they exist
      if tf.gfile.Exists(checkpoint_dir + "/game_res.pkl"):
        with tf.gfile.Open(checkpoint_dir + "/game_res.pkl",
                           "rb") as f:
          data_file = pickle.load(f)
          data_file["game_res"] = game_res
    
    for ep in logger.iter_bar(episodes=range(FLAGS.num_train_episodes)):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logger(message=f"{RED}{BOLD}Losses: {losses}{ENDC}")
        if FLAGS.evaluation_metric == "exploitability":
          expl = exploitability.exploitability(env.game, joint_avg_policy)
          logger(message=f"{OKGREEN}{BOLD}[{ep + 1}] Exploitability AVG {expl}{ENDC}")
          game_res.append(expl)
        elif FLAGS.evaluation_metric == "nash_conv":
          nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)
          logger(message=f"{OKGREEN}{BOLD}[{ep + 1}] NashConv {nash_conv}{ENDC}")
          game_res.append(nash_conv)
        elif FLAGS.evaluation_metric == "random_games":
          rand_eval = run_random_games(env.game, joint_avg_policy,
                                       FLAGS.num_eval_games)
          logger(message=f"{OKGREEN}{BOLD}[{ep + 1}] Random Games AVG {rand_eval}{ENDC}")
          game_res.append(rand_eval)
        else:
          raise ValueError(" ".join(
              ("Invalid evaluation metric, choose from",
               "'exploitability', 'nash_conv', 'random_games'.")))
        # nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)
        # logger(message=f"{OKGREEN}{BOLD}[{ep + 1}] NashConv {nash_conv}{ENDC}")
        # nash_res.append(nash_conv)
        if FLAGS.use_checkpoints:
          for agent in agents:
            agent.save(checkpoint_dir)
          data = {
              "game_res": game_res,
              "nash_res": nash_res,
              "num_train_episodes": FLAGS.num_train_episodes,
              "eval_every": FLAGS.eval_every,
              "num_eval_games": FLAGS.num_eval_games,
              "game_name": FLAGS.game_name,
              
          }
          with tf.gfile.Open(checkpoint_dir + "/game_res.pkl",
                              "wb") as f:
            pickle.dump(data, f)
        logger(message=f"{OKBLUE}----------------------------------------{ENDC}")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


def run_random_games(game, policy, num_games, player=None):
  """Runs random games and returns average score."""
  scores_as_p = [0., 0.]
  games_per_p = num_games if player else num_games // 2
  
  for _ in logger.iter_bar(Games=range(games_per_p)):
    if player:
      scores_as_p[player] += run_random_game(game, policy, player) / games_per_p
    else:
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
  # play_with_agent(None)
