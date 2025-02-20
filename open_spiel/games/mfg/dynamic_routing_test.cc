// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel::dynamic_routing {
namespace {

namespace testing = open_spiel::testing;

void TestLoad() {
  testing::LoadGameTest(
      "mfg_dynamic_routing(max_num_time_step=10,time_step_length=20.0"
      ",network_name=line)");
  auto game = LoadGame(
      "mfg_dynamic_routing(max_num_time_step=10,time_step_length=20.0"
      ",network_name=line)");
  auto state = game->NewInitialState();
  auto cloned = state->Clone();
  SPIEL_CHECK_EQ(state->ToString(), cloned->ToString());
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kMeanField);
  testing::ChanceOutcomesTest(*game);
}

void TestLoadWithParams() {
  testing::LoadGameTest(
      "mfg_dynamic_routing(max_num_time_step=10,time_step_length=20.0"
      ",network_name=line)");
  auto game = LoadGame(
      "mfg_dynamic_routing(max_num_time_step=10,time_step_length=20.0"
      ",network_name=line)");
  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->ObservationTensorShape().size(), 1);
  SPIEL_CHECK_EQ(game->ObservationTensorShape()[0],
                 game->NumDistinctActions() * 2 + game->MaxGameLength() + 2);
}

void TestWholeGameWithLineNetwork() {
  std::vector<double> distribution{1};
  auto game = LoadGame(
      "mfg_dynamic_routing(max_num_time_step=5,time_step_length=0.5,"
      "network_name=line)");
  auto state = game->NewInitialState();

  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->ToString(), "Before initial chance node.");
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  SPIEL_CHECK_EQ(state->ActionToString(0),
                 "Vehicle is assigned to population 0");
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kDefaultPlayerId);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=bef_O->O, waiting time=0, t=0_default, destination=D->aft_D"
      ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{3});
  state->ApplyAction(3);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=O->A, waiting time=-1, t=1_mean_field, destination=D->aft_D"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=O->A, waiting time=1, t=1_default, destination=D->aft_D"
      ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=O->A, waiting time=0, t=2_mean_field, destination=D->aft_D"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=O->A, waiting time=0, t=2_default, destination=D->aft_D"
      ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{1});
  state->ApplyAction(1);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->D, waiting time=-1, t=3_mean_field, destination=D->aft_D"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->D, waiting time=1, t=3_default, destination=D->aft_D"
      ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->D, waiting time=0, t=4_mean_field, destination=D->aft_D"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->D, waiting time=0, t=4_default, destination=D->aft_D"
      ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{2});
  state->ApplyAction(2);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Arrived at D->aft_D, with travel time 4.000000, t=5_terminal"
                 ", return=-2.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Arrived at D->aft_D, with travel time 4.000000, t=5_terminal"
                 ", return=-2.00");
}

void TestWholeGameWithBraessNetwork() {
  std::vector<double> distribution{1};
  auto game = LoadGame(
      "mfg_dynamic_routing(max_num_time_step=12,time_step_length=0.5,"
      "network_name=braess)");
  auto state = game->NewInitialState();

  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->ToString(), "Before initial chance node.");
  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  SPIEL_CHECK_EQ(state->ActionToString(0),
                 "Vehicle is assigned to population 0");
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kDefaultPlayerId);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=O->A, waiting time=0, t=0_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), (std::vector<Action>{1, 2}));
  state->ApplyAction(1);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->B, waiting time=-1, t=1_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=A->B, waiting time=3, t=1_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), (std::vector<Action>{0}));
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->B, waiting time=2, t=2_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=A->B, waiting time=2, t=2_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), (std::vector<Action>{0}));
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->B, waiting time=1, t=3_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=A->B, waiting time=1, t=3_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), (std::vector<Action>{0}));
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->B, waiting time=0, t=4_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=A->B, waiting time=0, t=4_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), (std::vector<Action>{3, 4}));
  state->ApplyAction(3);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=B->C, waiting time=-1, t=5_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=B->C, waiting time=0, t=5_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{5});
  state->ApplyAction(5);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=C->D, waiting time=-1, t=6_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=C->D, waiting time=3, t=6_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=C->D, waiting time=2, t=7_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=C->D, waiting time=2, t=7_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=C->D, waiting time=1, t=8_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=C->D, waiting time=1, t=8_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=C->D, waiting time=0, t=9_mean_field, destination=D->E"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=C->D, waiting time=0, t=9_default, destination=D->E"
                 ", return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{6});
  state->ApplyAction(6);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Arrived at D->E, with travel time 9.000000, t=10_mean_field"
                 ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Arrived at D->E, with travel time 9.000000, t=10_default, return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Arrived at D->E, with travel time 9.000000, t=11_mean_field, "
                 "return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Arrived at D->E, with travel time 9.000000, t=11_default, return=0.00");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{0});
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Arrived at D->E, with travel time 9.000000, t=12_terminal, "
                 "return=-4.50");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Arrived at D->E, with travel time 9.000000, t=12_terminal, "
                 "return=-4.50");

  SPIEL_CHECK_EQ(state->LegalActions(), std::vector<Action>{});
}

void TestPreEndedGameWithLineNetwork() {
  std::vector<double> distribution{1};
  auto game = LoadGame(
      "mfg_dynamic_routing(max_num_time_step=2,time_step_length=0.5,"
      "network_name=line)");
  auto state = game->NewInitialState();

  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_EQ(state->ToString(), "Before initial chance node.");
  SPIEL_CHECK_EQ(state->ActionToString(state->LegalActions()[0]),
                 "Vehicle is assigned to population 0");

  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kDefaultPlayerId);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=bef_O->O, waiting time=0, t=0_default, destination=D->aft_D"
      ", return=0.00");

  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=O->A, waiting time=-1, t=1_mean_field, destination=D->aft_D"
      ", return=0.00");

  state->UpdateDistribution(distribution);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=O->A, waiting time=1, t=1_default, destination=D->aft_D"
      ", return=0.00");

  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Arrived at O->A, with travel time 3.000000, t=2_terminal, return=-1.50");
}

void TestRandomPlayWithLineNetwork() {
  testing::RandomSimTest(
      *LoadGame("mfg_dynamic_routing(max_num_time_step=10,time_step_length=0.5,"
                "network_name=line,perform_sanity_checks=true)"),
      3);
}

void TestRandomPlayWithBraessNetwork() {
  testing::RandomSimTest(
      *LoadGame("mfg_dynamic_routing(max_num_time_step=10,time_step_length=0.5,"
                "network_name=braess,perform_sanity_checks=true)"),
      3);
}

// Test travel time update based on distribution is correct.
void TestCorrectTravelTimeUpdate() {
  auto game = LoadGame(
      "mfg_dynamic_routing(max_num_time_step=100,time_step_length=0.05,"
      "network_name=braess)");
  auto state = game->NewInitialState();

  SPIEL_CHECK_EQ(state->ToString(), "Before initial chance node.");
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=O->A, waiting time=0, t=0_default, destination=D->E"
                 ", return=0.00");
  SPIEL_CHECK_EQ(state->LegalActions(), (std::vector<Action>{1, 2}));
  state->ApplyAction(1);
  SPIEL_CHECK_EQ(
      state->ToString(),
      "Location=A->B, waiting time=-1, t=1_mean_field, destination=D->E"
      ", return=0.00");

  std::vector<double> distribution{1};
  state->UpdateDistribution({.5});
  // Waiting time (in unit of time) = 1.0 (free flow travel time on A->B) +
  //  .5 (% player on A->B) * 5 (num of players) / 5 (capacity on A->B) = 1.5
  // Waiting time (in time step) = 1.5 / 0.05 (time step lenght)
  //  - 1 (one time step for the current time running) = 29
  SPIEL_CHECK_EQ(state->ToString(),
                 "Location=A->B, waiting time=29, t=1_default, destination=D->E"
                 ", return=0.00");
}
}  // namespace
}  // namespace open_spiel::dynamic_routing

int main(int argc, char** argv) {
  open_spiel::dynamic_routing::TestLoad();
  open_spiel::dynamic_routing::TestLoadWithParams();
  open_spiel::dynamic_routing::TestWholeGameWithLineNetwork();
  open_spiel::dynamic_routing::TestWholeGameWithBraessNetwork();
  open_spiel::dynamic_routing::TestPreEndedGameWithLineNetwork();
  open_spiel::dynamic_routing::TestRandomPlayWithLineNetwork();
  open_spiel::dynamic_routing::TestRandomPlayWithBraessNetwork();
  open_spiel::dynamic_routing::TestCorrectTravelTimeUpdate();
}
