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
"""Approximate Best Response using Deep Q-Network."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pickle
import tqdm
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn


class ApproximateBestResponseDQN:

  def __init__(
      self,
      game,
      eval_id,
      eval_policy,
      checkpoint_dir="tmp/abr_dqn_test",
      save_every=int(1e5),
      num_train_episodes=int(1e6),
      eval_every=int(1e4),
      num_eval_games=int(1e4),
      hidden_layers_sizes=[128, 128],
      replay_buffer_capacity=int(1e5),
      batch_size=32,
  ):
    """
    Args:
      game: OpenSpiel game.
      eval_id: The id of the agent to evaluate.
      eval_policy: The policy to evaluate the agent.
      checkpoint_dir: Directory to save/load the agent models.
      save_every: Episode frequency at which the DQN agent models are saved.
      num_train_episodes: Number of training episodes.
      eval_every: Episode frequency at which the DQN agents are evaluated.
      num_eval_games: Number of games to evaluate the DQN agents.
      hidden_layers_sizes: Number of hidden units in the Q-Network MLP.
      replay_buffer_capacity: Size of the replay buffer.
      batch_size: Number of transitions to sample at each learning step.
    """
    self.env = rl_environment.Environment(game, use_tensor_observation=False)
    self.eval_id = eval_id
    self.br_id = 1 - self.eval_id
    self.eval_policy = eval_policy
    self.checkpoint_dir = checkpoint_dir
    self.save_every = save_every
    self.num_train_episodes = num_train_episodes
    self.eval_every = eval_every
    self.num_eval_games = num_eval_games
    self.hidden_layers_sizes = hidden_layers_sizes
    self.replay_buffer_capacity = replay_buffer_capacity
    self.batch_size = batch_size

  def _get_action_probabilities(self, info_state):
    a_p = self.eval_policy[info_state[self.eval_id]]
    return {a: p for a, p in a_p}

  def _get_action_from_pi(self, time_step):
    info_state = time_step.observations["info_state_str"]
    action_probs = self._get_action_probabilities(info_state)
    return np.random.choice(
        list(action_probs.keys()), p=list(action_probs.values()))

  def eval_against_pi(self, agent):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    sum_episode_rewards = 0.0
    for _ in range(self.num_eval_games):
      time_step = self.env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == self.eval_id:
          # act according to the evaluation policy
          action = self._get_action_from_pi(time_step)
        else:
          # act according to the training policy
          agent_output = agent.step(time_step, is_evaluation=True)
          action = agent_output.action
        time_step = self.env.step([action])
        episode_rewards += time_step.rewards[self.br_id]
      sum_episode_rewards += episode_rewards
    return sum_episode_rewards / self.num_eval_games

  def approximate_best_response(self):
    info_state_size = self.env.observation_spec()["info_state"][0]
    num_actions = self.env.action_spec()["num_actions"]

    with tf.Session() as sess:
      hidden_layers_sizes = [int(l) for l in self.hidden_layers_sizes]
      # pylint: disable=g-complex-comprehension
      agent = dqn.DQN(
          session=sess,
          player_id=self.br_id,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          hidden_layers_sizes=hidden_layers_sizes,
          replay_buffer_capacity=self.replay_buffer_capacity,
          batch_size=self.batch_size)
      sess.run(tf.global_variables_initializer())

      mean_rewards = []
      for ep in tqdm.tqdm(range(self.num_train_episodes)):
        if (ep + 1) % self.eval_every == 0 or ep in [
            0, self.num_train_episodes - 1
        ]:
          r_mean = self.eval_against_pi(agent)
          print("[%s] Mean episode rewards %s", ep + 1, r_mean)
          mean_rewards.append(r_mean)
        if (ep + 1) % self.save_every == 0:
          agent.save(self.checkpoint_dir)

        time_step = self.env.reset()
        while not time_step.last():
          player_id = time_step.observations["current_player"]
          # todo: add chance nodes
          if player_id == self.eval_id:
            # act according to the evaluation policy
            action = self._get_action_from_pi(time_step)
          else:
            # act according to the training policy
            agent_output = agent.step(time_step)
            action = agent_output.action
          time_step = self.env.step([action])

        # Episode is over, step all agents with final info state.
        agent.step(time_step)
      agent.save(self.checkpoint_dir)
      # save the mean rewards
      with open(self.checkpoint_dir + "/mean_rewards.pkl", "wb") as f:
        pickle.dump(mean_rewards, f)
      self._plot_rewards(mean_rewards)

  def _plot_rewards(self, mean_rewards):
    import matplotlib.pyplot as plt
    plt.plot(mean_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Mean episode rewards")
    plt.savefig(self.checkpoint_dir + "/mean_rewards.png")


if __name__ == "__main__":
  # game = "phantom_ttt_ir"
  # with open("tmp/phantom_ttt_p0_simplified_9a_0.1eps.pkl", "rb") as f:
  #   data = pickle.load(f)

  with open("../darkhex/darkhex/data/strategy_data/4x3_1_def/game_info.pkl",
            "rb") as f:
    data = pickle.load(f)
  num_cols = data["num_cols"]
  num_rows = data["num_rows"]
  game = f"dark_hex_ir(num_cols={num_cols},num_rows={num_rows},use_early_terminal=True)"

  evaluation_player = data["player"]
  policy = data["strategy"]
  abr = ApproximateBestResponseDQN(game, evaluation_player, policy)
  abr.approximate_best_response()
