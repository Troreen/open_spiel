import pyspiel

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
    # self.assertEqual(is_terminal, False)
    assert is_terminal == False
    # self.assertEqual(is_early_terminal, (True, 1))
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

test_dark_hex_early_terminal()
test_dark_hex_num_hidden_stones()