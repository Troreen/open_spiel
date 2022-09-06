// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/cheat.h"

#include <algorithm>
#include <map>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cheat {
namespace {

const GameType kGameType{
    /*short_name=*/"cheat",
    /*long_name=*/"Cheat",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CheatGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CheatGame::CheatGame(const GameParameters& params)
    : Game(kGameType, params) {}

CheatState::CheatState(std::shared_ptr<const Game> game)
    : State(game) {}

std::string CheatState::ActionToString(Player player, Action action) const {
  // convert the action to cards -> 
  return CardString(action);
}

std::string CheatState::ToString() const {
  absl::StrAppend(&rv, FormatDeal());
  if (!passed_cards_[0].empty()) absl::StrAppend(&rv, FormatPass());
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::string CheatState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = "Hand: \n";
  auto cards = FormatHand(player, /*mark_voids=*/true);
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (!passed_cards_[player].empty()) absl::StrAppend(&rv, FormatPass(player));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

// std::array<std::string, kNumSuits> CheatState::FormatHand(
//     int player, bool mark_voids) const {
//   // Current hand, except in the terminal state when we use the original hand
//   // to enable an easy review of the whole deal.
//   auto deal = IsTerminal() ? initial_deal_ : player_hand_;
//   std::array<std::string, kNumSuits> cards;
//   for (int suit = 0; suit < kNumSuits; ++suit) {
//     cards[suit].push_back(kSuitChar[suit]);
//     cards[suit].push_back(' ');
//     for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
//       if (player == deal[Card(Suit(suit), rank)]) {
//         cards[suit].push_back(kRankChar[rank]);
//         is_void = false;
//       }
//     }
//     if (is_void && mark_voids) absl::StrAppend(&cards[suit], "none");
//   }
//   return cards;
// }

// std::string CheatState::FormatDeal() const {
//   std::string s;
//   std::array<std::array<std::string, kNumSuits>, kNumPlayers> cards;
//   for (auto player : {kFirstPlayer, kSecondPlayer})
//     cards[player] = FormatHand(player, /*mark_voids=*/false);
//   constexpr int kColumnWidth = 8;
//   std::string padding(kColumnWidth, ' ');
//   for (int suit = kNumSuits - 1; suit >= 0; --suit)
//     absl::StrAppend(&rv, padding, cards[kNorth][suit], "\n");
//   for (int suit = kNumSuits - 1; suit >= 0; --suit)
//     absl::StrAppend(&rv, absl::StrFormat("%-8s", cards[kWest][suit]), padding,
//                     cards[kEast][suit], "\n");
//   for (int suit = kNumSuits - 1; suit >= 0; --suit)
//     absl::StrAppend(&rv, padding, cards[kSouth][suit], "\n");
//   return s;
// }

// std::string CheatState::FormatPass() const {
//   std::string rv = "\n\nPassed Cards:";
//   for (int player = 0; player < kNumPlayers; ++player) {
//     absl::StrAppend(&rv, "\n", DirString(player), ": ");
//     for (int card : passed_cards_[player]) {
//       absl::StrAppend(&rv, CardString(card), " ");
//     }
//   }
//   // Cards are not received until all players have completed passing.
//   // West is the last player to pass.
//   if (passed_cards_[static_cast<int>(kWest)].size() == kNumCardsInPass) {
//     absl::StrAppend(&rv, "\n\nReceived Cards:");
//     for (int receiver = 0; receiver < kNumPlayers; ++receiver) {
//       absl::StrAppend(&rv, "\n", DirString(receiver), ": ");
//       int passer =
//           (receiver + kNumPlayers - static_cast<int>(pass_dir_)) % kNumPlayers;
//       for (int card : passed_cards_[passer]) {
//         absl::StrAppend(&rv, CardString(card), " ");
//       }
//     }
//   }
//   absl::StrAppend(&rv, "\n");
//   return rv;
// }

// std::string CheatState::FormatPass(Player player) const {
//   std::string rv = "\nPassed Cards: ";
//   std::vector<int> passed_cards = passed_cards_[player];
//   // Sort cards because players don't have access to the order in which the
//   // cards were selected to be passed. Knowing the order could allow for
//   // information leakage.
//   absl::c_sort(passed_cards);
//   for (int card : passed_cards) absl::StrAppend(&rv, CardString(card), " ");
//   // Cards are not received until all players have completed passing.
//   // West is the last player to pass.
//   if (passed_cards_[static_cast<int>(kWest)].size() == kNumCardsInPass) {
//     absl::StrAppend(&rv, "\n\nReceived Cards: ");
//     int passing_player =
//         (player + kNumPlayers - static_cast<int>(pass_dir_)) % kNumPlayers;
//     std::vector<int> received_cards = passed_cards_[passing_player];
//     absl::c_sort(received_cards);
//     for (int card : received_cards) absl::StrAppend(&rv, CardString(card), " ");
//   }
//   absl::StrAppend(&rv, "\n");
//   return rv;
// }

// std::string CheatState::FormatPlay() const {
//   SPIEL_CHECK_GT(num_cards_played_, 0);
//   std::string rv = "\nTricks:";
//   absl::StrAppend(&rv, "\nN  E  S  W  N  E  S");
//   for (int i = 0; i <= (num_cards_played_ - 1) / kNumPlayers; ++i) {
//     absl::StrAppend(&rv, "\n", std::string(3 * tricks_[i].Leader(), ' '));
//     for (auto card : tricks_[i].Cards()) {
//       absl::StrAppend(&rv, CardString(card), " ");
//     }
//   }
//   return rv;
// }

// std::string CheatState::FormatPoints() const {
//   std::string rv;
//   absl::StrAppend(&rv, "\n\nPoints:");
//   for (int i = 0; i < kNumPlayers; ++i)
//     absl::StrAppend(&rv, "\n", DirString(i), ": ", points_[i]);
//   return rv;
// }

void CheatState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0);
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  auto ptr = values.begin();
  // Current hand
  for (int i = 0; i < kNumCards; ++i)
    if (player_hand_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Cards claimed
  for (int i = 0; i < kNumCards; ++i)
    if (card_claimed_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Cards seen
  for (int i = 0; i < kNumCards; ++i)
    if (card_seen_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Action history
  for (int i = 0; i < kNumPlayers; ++i) {
    for (int j = 0; j < kNumCards; ++j) {
      ptr[i] = action_history_[i][j];
    }
    ptr += kNumCards;
  }
  SPIEL_CHECK_EQ(ptr, values.end());
}

std::vector<Action> CheatState::LegalActions() const {
  std::vector<Action> legal_actions;
  // If initial move, deal cards - chance node
  if (num_cards_dealt_ == 0) {
    // Deal 7 random cards to each player
    for (int i = 0; i < kNumPlayers; ++i) {
      for (int j = 0; j < kNumInitCardsPerPlayer; ++j) {
        legal_actions.push_back(kDealAction);
      }
    }
    legal_actions.reserve(kNumCards - num_cards_dealt_);
    for (int i = 0; i < kNumCards; ++i) {
      if (!player_hand_[i].has_value()) legal_actions.push_back(i);
    }
    SPIEL_CHECK_GT(legal_actions.size(), 0);
    return legal_actions;
  }
  legal_actions.reserve(kNumTricks - num_cards_played_ / kNumPlayers);

  // Check if we can follow suit.
  if (num_cards_played_ % kNumPlayers != 0) {
    auto suit = CurrentTrick().LedSuit();
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (player_hand_[Card(suit, rank)] == current_player_) {
        legal_actions.push_back(Card(suit, rank));
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Special rules apply to the first card played.
  // Must play 2C unless option to play any club is enabled.
  if (num_cards_played_ == 0) {
    SPIEL_CHECK_TRUE(player_hand_[Card(Suit::kClubs, 0)] == current_player_);
    legal_actions.push_back(Card(Suit::kClubs, 0));
    if (can_lead_any_club_) {
      for (int rank = 1; rank < kNumCardsPerSuit; ++rank) {
        if (player_hand_[Card(Suit::kClubs, rank)] == current_player_) {
          legal_actions.push_back(Card(Suit::kClubs, rank));
        }
      }
    }
    return legal_actions;
  }

  // Special rules apply to the first trick.
  // By default, cannot play hearts or QS on first trick.
  if (no_pts_on_first_trick_ && num_cards_played_ < kNumPlayers) {
    for (int card = 0; card < kNumCards; ++card) {
      if (player_hand_[card] == current_player_ && card != Card(Suit::kSpades, 10) &&
          CardSuit(card) != Suit::kCheat) {
        legal_actions.push_back(card);
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Player must lead. By default, cannot lead hearts until broken.
  if (num_cards_played_ % kNumPlayers == 0) {
    for (int card = 0; card < kNumCards; ++card) {
      if (player_hand_[card] == current_player_) {
        if (CardSuit(card) != Suit::kCheat || hearts_broken_) {
          legal_actions.push_back(card);
        }
      }
    }
    // Don't force player into leading the QS when hearts have not
    // been broken.
    if (can_lead_hearts_instead_of_qs_ && legal_actions.size() == 1 &&
        legal_actions[0] == Card(Suit::kSpades, 10)) {
      legal_actions.pop_back();
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Otherwise, we can play any of our cards.
  for (int card = 0; card < kNumCards; ++card) {
    if (player_hand_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> CheatState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  if (history_.empty()) {
    outcomes.reserve(kNumPlayers);
    const double p = 1.0 / kNumPlayers;
    for (int dir = 0; dir < kNumPlayers; ++dir) {
      outcomes.emplace_back(dir, p);
    }
    return outcomes;
  }
  int num_cards_remaining = kNumCards - num_cards_dealt_;
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / num_cards_remaining;
  for (int card = 0; card < kNumCards; ++card) {
    if (!player_hand_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void CheatState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kPass:
      return ApplyPassAction(action);
    case Phase::kPlay:
      return ApplyPlayAction(action);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

void CheatState::ApplyDealAction(int card) {
  player_hand_[card] = num_cards_dealt_ % kNumPlayers;
  ++num_cards_dealt_;
  if (num_cards_dealt_ == kNumCards) {
    // Preserve the initial deal for easy retrieval
    initial_deal_ = player_hand_;
    if (pass_dir_ == PassDir::kNoPass) {
      phase_ = Phase::kPlay;
      // Play starts with the holder of the 2C
      current_player_ = player_hand_[Card(Suit::kClubs, 0)].value();
    } else {
      phase_ = Phase::kPass;
      current_player_ = 0;
    }
  }
}

void CheatState::ApplyPassAction(int card) {
  passed_cards_[current_player_].push_back(card);
  player_hand_[card] = absl::nullopt;
  if (passed_cards_[current_player_].size() % kNumCardsInPass == 0)
    ++current_player_;
  if (current_player_ == kNumPlayers) {
    // Players have completed passing. Now let's distribute the passed cards.
    for (int player = 0; player < kNumPlayers; ++player) {
      for (int card : passed_cards_[player]) {
        player_hand_[card] = (player + static_cast<int>(pass_dir_)) % kNumPlayers;
      }
    }
    phase_ = Phase::kPlay;
    // Play starts with the holder of the 2C
    current_player_ = player_hand_[Card(Suit::kClubs, 0)].value();
  }
}

void CheatState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(player_hand_[card] == current_player_);
  player_hand_[card] = absl::nullopt;
  if (num_cards_played_ % kNumPlayers == 0) {
    CurrentTrick() = Trick(current_player_, card, jd_bonus_);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  // Check if action breaks hearts.
  if (CardSuit(card) == Suit::kCheat) hearts_broken_ = true;
  if (qs_breaks_hearts_ && card == Card(Suit::kSpades, 10))
    hearts_broken_ = true;
  // Update player and point totals.
  Trick current_trick = CurrentTrick();
  ++num_cards_played_;
  if (num_cards_played_ % kNumPlayers == 0) {
    current_player_ = current_trick.Winner();
    points_[current_player_] += current_trick.Points();
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
  }
  if (num_cards_played_ == kNumCards) {
    phase_ = Phase::kGameOver;
    current_player_ = kTerminalPlayerId;
    ComputeScore();
  }
}

Player CheatState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  return current_player_;
}

void CheatState::ComputeScore() {
  SPIEL_CHECK_TRUE(IsTerminal());
  // Did anyone shoot the moon?
  Player potential_shooter = kInvalidPlayer;
  bool moon_shot = true;
  for (int i = 0; i < kNumTricks; ++i) {
    int points = tricks_[i].Points();
    // JD not required to shoot the moon.
    if (points != 0 && points != kPointsForJD) {
      // This trick must be taken by the shooter.
      if (potential_shooter == kInvalidPlayer) {
        potential_shooter = tricks_[i].Winner();
      } else if (potential_shooter != tricks_[i].Winner()) {
        moon_shot = false;
        break;
      }
    }
  }
  // Shooting the moon sets the shooter's points to 0, and adds 26 pts to each
  // opponent's score.
  if (moon_shot) {
    for (int i = 0; i < kNumPlayers; ++i) {
      points_[i] += (i == potential_shooter) ? -kTotalPositivePoints
                                             : kTotalPositivePoints;
    }
  }
  // Did anyone avoid taking any tricks?
  if (avoid_all_tricks_bonus_ && !moon_shot) {
    std::vector<int> tricks_taken(kNumPlayers, 0);
    for (int i = 0; i < kNumTricks; ++i) {
      tricks_taken[tricks_[i].Winner()] += 1;
    }
    for (int i = 0; i < kNumPlayers; ++i) {
      if (tricks_taken[i] == 0) points_[i] += kAvoidAllTricksBonus;
    }
  }
}

std::vector<double> CheatState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(kNumPlayers, 0.0);
  }
  std::vector<double> returns = points_;
  for (int i = 0; i < returns.size(); ++i)
    returns[i] = kTotalPositivePoints - returns[i];
  return returns;
}

absl::optional<Player> CheatState::Played(int card) const {
  if (phase_ == Phase::kPlay && !player_hand_[card].has_value()) {
    Player p = *(initial_deal_[card]);
    // check if they kept the card or not
    auto it = std::find(passed_cards_[p].begin(), passed_cards_[p].end(), card);
    if (it != passed_cards_[p].end()) {
      p = (p + static_cast<int>(pass_dir_)) % kNumPlayers;
    }
    return p;
  }

  return absl::nullopt;
}

bool CheatState::KnowsLocation(Player player, int card) const {
  bool dealt, received, played, two_clubs;
  dealt = initial_deal_[card] == player;
  int pass_dir = static_cast<int>(pass_dir_);
  Player recv_from = (player + kNumPlayers - pass_dir) % kNumPlayers;
  auto it = std::find(passed_cards_[recv_from].begin(),
                      passed_cards_[recv_from].end(), card);
  received = it != passed_cards_[recv_from].end() && phase_ == Phase::kPlay;
  played = !player_hand_[card].has_value() && phase_ == Phase::kPlay;
  two_clubs = card == Card(Suit::kClubs, 0) && phase_ == Phase::kPlay;
  return dealt || received || played || two_clubs;
}

// Does not account for void suit information exposed by other players during
// the play phase
// std::unique_ptr<State> CheatState::ResampleFromInfostate(
//     int player_id, std::function<double()> rng) const {
//   std::unique_ptr<State> clone = game_->NewInitialState();
//   Action pass_dir = static_cast<int>(pass_dir_);
//   clone->ApplyAction(pass_dir);

//   // start by gathering all public and private info known to player_id to
//   // simplify the logic for applying deal / pass actions
//   // first thing we know is the player's entire hand
//   std::vector<int> initial_hand;
//   for (int card = 0; card < kNumCards; card++) {
//     if (initial_deal_[card] == player_id) initial_hand.push_back(card);
//   }

//   // collect cards that have been revealed through the play phase
//   std::vector<std::vector<int>> play_known(kNumPlayers);
//   if (phase_ == Phase::kPlay) {
//     for (int card = 0; card < kNumCards; card++) {
//       absl::optional<Player> p = Played(card);
//       if (p && *p != player_id) {
//         play_known[*p].push_back(card);
//       }
//     }
//   }

//   // the two of clubs is also known once a player has played first
//   absl::optional<Player> two_clubs_holder = player_hand_[Card(Suit::kClubs, 0)];
//   if (phase_ == Phase::kPlay && two_clubs_holder) {
//     play_known[*two_clubs_holder].push_back(Card(Suit::kClubs, 0));
//   }

//   // set up pass cards greedily using known cards so that passes are
//   // consistent
//   // this shouldn't affect the distribution of resampled states much because
//   // we have no way to model unobserved opponent pass actions anyway
//   std::vector<std::vector<int>> pass_actions(kNumPlayers);
//   for (Player p = 0; p < kNumPlayers; p++) {
//     for (int pass_num = 0; pass_num < passed_cards_[p].size(); pass_num++) {
//       if (p == player_id) {
//         pass_actions[p].push_back(passed_cards_[p][pass_num]);
//       } else {
//         Player pass_to = (p + pass_dir) % kNumPlayers;
//         // once the play phase has started, player_id knows the cards that were
//         // passed to them
//         if (phase_ == Phase::kPlay && pass_to == player_id) {
//           pass_actions[p].push_back(passed_cards_[p][pass_num]);
//         } else if (pass_num < play_known[pass_to].size()) {
//           pass_actions[p].push_back(play_known[pass_to][pass_num]);
//         }
//       }
//     }
//   }

//   // at this point we have all the information we need about which card
//   // locations are known to player_id, so we can start applying deal actions
//   Player deal_to, pass_to, recv_from;
//   int card_num;
//   std::vector<bool> dealt(kNumCards, false);
//   std::vector<int> known_dealt_counter(kNumPlayers, 0);
//   for (int num_dealt = 0; num_dealt < kNumCards; num_dealt++) {
//     card_num = num_dealt / kNumPlayers;
//     deal_to = num_dealt % kNumPlayers;
//     pass_to = (deal_to + pass_dir) % kNumPlayers;
//     recv_from = (deal_to + kNumPlayers - pass_dir) % kNumPlayers;
//     Action action = kInvalidAction;
//     // deal out the pass moves first so those constraints are satisfied
//     if (card_num < pass_actions[deal_to].size()) {
//       action = pass_actions[deal_to][card_num];
//     } else {
//       // now try to find any cards that deal_to has shown they have that
//       // haven't already been allocated as a pass action for recv_from and have
//       // not already been dealt
//       auto& known = (deal_to == player_id) ? initial_hand : play_known[deal_to];
//       while ((action == kInvalidAction || dealt[action]) &&
//              known_dealt_counter[deal_to] < known.size()) {
//         action = known[known_dealt_counter[deal_to]];
//         auto it = std::find(pass_actions[recv_from].begin(),
//                             pass_actions[recv_from].end(), action);
//         if (it != pass_actions[recv_from].end()) action = kInvalidAction;
//         known_dealt_counter[deal_to]++;
//       }
//     }

//     // all known card constraints for to_deal have been satisfied, so we can
//     // deal them a random card that does not violate other player constraints
//     while (action == kInvalidAction) {
//       Action candidate = SampleAction(clone->ChanceOutcomes(), rng()).first;
//       if (!KnowsLocation(player_id, candidate)) {
//         action = candidate;
//         // we can also use this card as a pass action later because its
//         // location is unknown
//         if (pass_actions[deal_to].size() < passed_cards_[deal_to].size()) {
//           pass_actions[deal_to].push_back(action);
//         }
//       }
//     }

//     clone->ApplyAction(action);
//     dealt[action] = true;
//   }

//   // now handle the pass phase
//   if (pass_dir_ != PassDir::kNoPass) {
//     for (Player to_move = 0; to_move < kNumPlayers; to_move++) {
//       SPIEL_CHECK_EQ(pass_actions[to_move].size(),
//                      passed_cards_[to_move].size());
//       pass_to = (to_move + pass_dir) % kNumPlayers;
//       for (int passes = 0; passes < passed_cards_[to_move].size(); passes++) {
//         Action action = kInvalidAction;
//         if (to_move == player_id || pass_to == player_id) {
//           // player_id knows exactly which cards were passed by player_id and
//           // the player who passed cards to player_id
//           action = passed_cards_[to_move][passes];
//         } else {
//           action = pass_actions[to_move][passes];
//         }
//         clone->ApplyAction(action);
//       }
//     }
//   }

//   // given that we should now have a state consistent with the public actions
//   // and player_id's private cards, we can just copy the action sequence in
//   // the play phase
//   int play_start_index = kNumCards + 1;
//   if (pass_dir_ != PassDir::kNoPass)
//     play_start_index += kNumPlayers * kNumCardsInPass;
//   for (size_t i = play_start_index; i < history_.size(); i++) {
//     clone->ApplyAction(history_.at(i).action);
//   }

//   SPIEL_CHECK_EQ(FullHistory().size(), clone->FullHistory().size());
//   SPIEL_CHECK_EQ(InformationStateString(player_id),
//                  clone->InformationStateString(player_id));
//   return clone;
// }

}  // namespace hearts
}  // namespace open_spiel
