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
"""Tests for open_spiel.python.algorithms.dqn."""

import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
import pyspiel

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()


def test_run_dark_hex_conv2d(self):
  num_rows = 3
  num_cols = 3
  env = rl_environment.Environment(
      "dark_hex_ir", num_rows=num_rows, num_cols=num_cols)

  state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  with self.session() as sess:
    agents = [
        dqn.DQN(  # pylint: disable=g-complex-comprehension
            sess,
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[16],
            replay_buffer_capacity=10,
            batch_size=5,
            model_type="conv2d",
            input_shape=(3, num_rows, num_cols),
            conv_layer_info=[{
                'filters': 512
            }]) for player_id in [0, 1]
    ]
    sess.run(tf.global_variables_initializer())
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      current_agent = agents[current_player]
      agent_output = current_agent.step(time_step)
      time_step = env.step([agent_output.action])

    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  tf.test.main()
