import gym
from gymhearts import env as hearts_env
from gymhearts import strategy
from gymhearts.evaluator import Evaluator

from treys import Deck
from treys import Card

from .completeplay import CompletePlayStrategy
from .mcts import MCTS


class MCTSPlayStrategy(strategy.IStrategy):
    
    def __init__(self):
        self._evaluator = Evaluator()
        deck = Deck()
        self._available_cards = deck.draw(52)

    def move(self, observation):
        pass 

    def watch(self, observation, info):
        if info['done'] is True:
            pass
        elif info['is_new_round'] is True:
            deck = Deck()
            self._available_cards = deck.draw(52)
        else:
            played_card = info['action']
            self._available_cards.remove(played_card)
        

