// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/games_dark_hex.h"

#include "open_spiel/games/dark_hex.h"
#include "open_spiel/spiel.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::State;
using open_spiel::dark_hex::DarkHexState;
using open_spiel::dark_hex::ImperfectRecallDarkHexState;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(DarkHexState);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(ImperfectRecallDarkHexState);

void open_spiel::init_pyspiel_games_dark_hex(py::module& m) {
  py::classh<DarkHexState, State>(m, "DarkHexState")
      .def("num_hidden_stones", &DarkHexState::NumHiddenStones)
      // Pickle support
      .def(py::pickle(
          [](const DarkHexState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
              game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<DarkHexState*>(
              game_and_state.second.release());
          }));
  py::classh<ImperfectRecallDarkHexState, State>(m, "ImperfectRecallDarkHexState")
      .def("is_early_terminal", &ImperfectRecallDarkHexState::IsEarlyTerminal)
      .def("num_hidden_stones", &DarkHexState::NumHiddenStones)
      // Pickle support
      .def(py::pickle(
          [](const ImperfectRecallDarkHexState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
              game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ImperfectRecallDarkHexState*>(
              game_and_state.second.release());
          }));
}
