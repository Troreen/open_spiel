from abc import ABC, abstractmethod

class Player(ABC):
    """
    Structure for a player
    """
    def __init__(self, p_type, path):
        self.p_type = p_type
        self.path = path

    @abstractmethod
    def read_data(self):
        """
        Reads the data from the file.
        Changes depending on the player type.
        """
        pass

    @abstractmethod
    def get_action(self, state):
        """
        Returns the action for the current state.
        """
        pass

    def get_action_probabilities(self, state):
        """
        Returns the action probabilities for the current state.
        """
        raise NotImplementedError("get_action_probabilities() not implemented")
