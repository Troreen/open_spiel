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
flags.DEFINE_integer("num_rows", 3, "Number of rows.")
flags.DEFINE_integer("num_cols", 3, "Number of cols.")
flags.DEFINE_integer("num_players", 2, "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(2e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", int(2e4),
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("num_eval_games", int(6e3),
                     "Number of evaluation games when running random_games evaluator.")
flags.DEFINE_list("hidden_layers_sizes", [
    126
], "Number of hidden units to use in each layer of the avg-net and Q-net.")
flags.DEFINE_list("conv_layer_info", [
    # '{"filters": 1024, "kernel_size": 3, "strides": 1, "padding": "same", "max_pool": 2}',
    # '{"filters": 512, "kernel_size": 2, "strides": 1, "padding": "same", "max_pool": 1}',
    '{"filters": 256, "kernel_size": 2, "strides": 1, "padding": "same", "max_pool": 0}',
], "Convolutional layers information. Each layer is a dictionary with the following keys: " +
   "filters (int), kernel_size (int), strides (int), padding (str), max_pool (int). " +
   "If max_pool is 0, no max pooling is used. If max_pool is a positive integer, max " +
   "pooling is used with the given pool size.")
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
flags.DEFINE_string("checkpoint_dir", "tmp/nfsp_test_3x3_sresnet_ir_checkpoints",
                    "Directory to save/load the agent.")


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
  
  env_configs = {"num_rows": FLAGS.num_rows, "num_cols": FLAGS.num_cols}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  # Parsing conv_layer_info
  conv_layer_info = []
  for layer_info in FLAGS.conv_layer_info:
    conv_layer_info.append(json.loads(layer_info))
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

  if FLAGS.use_checkpoints:
    # Create the folder if it doesn't exist.
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(
            sess,
            idx,
            info_state_size,
            num_actions,
            hidden_layers_sizes,
            conv_layer_info=conv_layer_info,
            # model_type='mlp',
            model_type=FLAGS.model_type,
            input_shape=(3, FLAGS.num_rows, FLAGS.num_cols),
            **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy, FLAGS.num_players)

    sess.run(tf.global_variables_initializer())

    game_res = []
    rand_res = []
    if FLAGS.use_checkpoints:
      for agent in agents:
        if agent.has_checkpoint(FLAGS.checkpoint_dir):
          agent.restore(FLAGS.checkpoint_dir)
      # load the random game results if they exist
      if tf.gfile.Exists(FLAGS.checkpoint_dir + "/game_res.pkl"):
        with tf.gfile.Open(FLAGS.checkpoint_dir + "/game_res.pkl",
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
        # elif FLAGS.evaluation_metric == "random_games":
        #   rand_eval = run_random_games(env.game, joint_avg_policy,
        #                                FLAGS.num_eval_games)
        #   logger(message=f"{OKGREEN}{BOLD}[{ep + 1}] Random Games AVG {rand_eval}{ENDC}")
        #   game_res.append(rand_eval)
        rand_eval = run_random_games(env.game, joint_avg_policy,
                                       FLAGS.num_eval_games)
        rand_res.append(rand_eval)
        # else:
        #   raise ValueError(" ".join(
        #       ("Invalid evaluation metric, choose from",
        #        "'exploitability', 'nash_conv', 'random_games'.")))
        if FLAGS.use_checkpoints:
          for agent in agents:
            agent.save(FLAGS.checkpoint_dir)
          # Save the random game results
          data = {
              "game_res": game_res,
              "rand_res": rand_res,
              "num_train_episodes": FLAGS.num_train_episodes,
              "eval_every": FLAGS.eval_every,
              "num_eval_games": FLAGS.num_eval_games,
          }
          with tf.gfile.Open(FLAGS.checkpoint_dir + "/game_res.pkl",
                              "wb") as f:
            pickle.dump(data, f)
        logger(message=f"{OKBLUE}_____________________________________________{ENDC}")

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


def play_with_agent(unused_argv):
  def_values = {
    'game_name': 'dark_hex_ir',
    'num_rows': 4,
    'num_cols': 3,
    'num_players': 2,
    'num_train_episodes': int(2e7),
    'eval_every': int(2e5),
    'num_eval_games': int(2e4),
    'hidden_layers_sizes': [512, 256, 126],
    'conv_layer_info': [
      '{"filters": 1024, "kernel_size": 3, "strides": 1, "padding": "same", "max_pool": 2}',
      '{"filters": 512, "kernel_size": 2, "strides": 1, "padding": "same", "max_pool": 1}',
      '{"filters": 256, "kernel_size": 2, "strides": 1, "padding": "same", "max_pool": 0}',
    ],
    'replay_buffer_capacity': int(2e5),
    'reservoir_buffer_capacity': int(2e6),
    'min_buffer_size_to_learn': 1000,
    'anticipatory_param': 0.1,
    'batch_size': 128,
    'learn_every': 64,
    'rl_learning_rate': 0.01,
    'sl_learning_rate': 0.01,
    'optimizer_str': 'sgd',
    'loss_str': 'mse',
    'model_type': 'resnet',
    'dropout_rate': 0.2,
    'use_batch_norm': 'True',
    'update_target_network_every': 19200,
    'discount_factor': 1.0,
    'epsilon_decay_duration': int(20e6),
    'epsilon_start': 0.06,
    'epsilon_end': 0.001,
    'evaluation_metric': 'random_games',
    'use_checkpoints': True,
    'checkpoint_dir': 'tmp/nfsp_test_4x3_resnet_checkpoints',
  }
  logger(message=f"{OKBLUE}Loading game {def_values['game_name']}{ENDC}")
  game = def_values['game_name']
  num_players = def_values['num_players']
  
  env_configs = {"num_rows": def_values['num_rows'], "num_cols": def_values['num_cols']}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in def_values['hidden_layers_sizes']]
  # Parsing conv_layer_info
  conv_layer_info = []
  for layer_info in def_values['conv_layer_info']:
    conv_layer_info.append(json.loads(layer_info))
  kwargs = {
      "replay_buffer_capacity": def_values['replay_buffer_capacity'],
      "reservoir_buffer_capacity": def_values['reservoir_buffer_capacity'],
      "min_buffer_size_to_learn": def_values['min_buffer_size_to_learn'],
      "anticipatory_param": def_values['anticipatory_param'],
      "batch_size": def_values['batch_size'],
      "learn_every": def_values['learn_every'],
      "rl_learning_rate": def_values['rl_learning_rate'],
      "sl_learning_rate": def_values['sl_learning_rate'],
      "optimizer_str": def_values['optimizer_str'],
      "loss_str": def_values['loss_str'],
      "update_target_network_every": def_values['update_target_network_every'],
      "discount_factor": def_values['discount_factor'],
      "epsilon_decay_duration": def_values['epsilon_decay_duration'],
      "epsilon_start": def_values['epsilon_start'],
      "epsilon_end": def_values['epsilon_end'],
      "use_batch_norm": def_values['use_batch_norm'],
      "dropout_rate": def_values['dropout_rate'],
  }

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(
            sess,
            idx,
            info_state_size,
            num_actions,
            hidden_layers_sizes,
            conv_layer_info=conv_layer_info,
            model_type=def_values['model_type'],
            input_shape=(3, def_values['num_rows'], def_values['num_cols']),
            **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy, def_values['num_players'])

    sess.run(tf.global_variables_initializer())

    # Load the checkpoints from the checkpoint directory
    for agent in agents:
      agent.restore(def_values['checkpoint_dir'])
          
    # Play against the average policy
    game = pyspiel.load_game('dark_hex_ir(num_rows=4,num_cols=3)')
    
    player = 0
    num_games = 10
    wins = [0] * num_players
    for n in range(num_games):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.current_player() == player:
          print(state.information_state_string(player))
          # print(state)
          action = int(input("Enter action: ").strip())
        else:
          policy_dict = joint_avg_policy.action_probabilities(state)
          # print(f"{policy_dict}")
          policy = [policy_dict.get(a, 0.0) for a in range(game.num_distinct_actions())]
          policy = np.array(policy)
          action = np.random.choice(np.arange(len(policy)), p=policy)
          # print(f"{state.current_player()}'s policy: {policy}")
          # get the max probability action]
          # action = np.argmax(policy)
        p_ = state.current_player()
        state.apply_action(action)
        # print(f"{p_} plays {action}")
      print(f"{p_} wins!")
      print(state)
      wins[p_] += 1
    print(f"\033[1;32mPlayer wins {wins[0]}\033[0m | \033[1;31m{wins[1]} Opponent wins\033[0m") 
      

if __name__ == "__main__":
  app.run(main)
  # play_with_agent(None)
