import math
import copy
import random
import hashlib

import gym
from gymhearts import env as hearts_env
from gymhearts import strategy
from gymhearts.evaluator import Evaluator

from treys import Deck
from treys import Card

from .completeplay import CompletePlayStrategy
from .mcts import MCTS
from .mcts import Node
from .mcts import IState


class HeartState(IState):

    def __init__(self, observation, info):
        if 'available_cards' not in info:
            deck = Deck()
            info['available_cards'] = deck.draw(52)
            for c in observation['hand_cards']:
                info['available_cards'].remove(c)
        self._observation = observation
        self._info = info
        self._evaluator = Evaluator()
        self._number_of_children = 1
        number_of_available_cards = len(info['available_cards'])
        self.num_moves = 1
        for n in range(number_of_available_cards, number_of_available_cards-3, -1):
            self.num_moves *= n
        self.num_moves *= len(observation['hand_cards'])

    def next_state(self):
        next_observation = copy.deepcopy(self._observation)
        next_info = copy.deepcopy(self._info)
        trick = next_observation['trick']
        my_player_id = next_info['my_player_id']
        number_of_players = next_observation['number_of_players']

        # generate valid hand cards
        valid_hand_cards = next_observation['hand_cards']

        # generate next playing cards
        playing_cards = random.sample(next_info['available_cards'], number_of_players-1)
        for c in playing_cards:
            next_info['available_cards'].remove(c)
        my_playing_card = random.choice(valid_hand_cards)
        next_observation['hand_cards'].remove(my_playing_card)
        playing_cards.insert(my_player_id, my_playing_card)

        # update observation and info
        playing_ids = range(0, number_of_players)
        looser_score, looser_player_id = self._evaluator.evaluate(playing_cards, playing_ids)
        if looser_player_id != my_player_id:
            next_observation['scores'][my_player_id] += looser_score
        next_observation['trick'] += 1
        next_observation['playing_cards'] = playing_cards
        return HeartState(next_observation, next_info)
    
    def get_action_card(self):
        my_player_id = self._info['my_player_id']
        return self._observation['playing_cards'][my_player_id]

    def terminal(self):
        return self._observation['trick'] == 13
    
    def reward(self):
        my_player_id = self._info['my_player_id']
        r = 1.0 - self._observation['scores'][my_player_id] / 26
        return r
    
    def __hash__(self):
        return int(hashlib.md5(
            (str(self._observation['trick']) + ":" + str(self._info['available_cards']) + ":" + str(self._observation['hand_cards'])).encode('utf-8')
        ).hexdigest(), 16)
    
    def __eq__(self,other):
        if hash(self) == hash(other):
            return True
        return False
    
    def __repr__(self):
        return "Score: %d, Playing Cards: %s "% (self._observation['scores'], self._observation['playing_cards'])


class MCTSPlayStrategy(strategy.IStrategy):
    
    def __init__(self, budget, my_player_id):
        self._my_player_id = my_player_id
        self._evaluator = Evaluator()
        self._mcts = MCTS(budget)
        self._current_node = None

    def move(self, observation):
        if self._current_node is None:
            new_info = {}
            new_info['my_player_id'] = self._my_player_id
            self._current_node = Node(HeartState(observation, new_info))
        best_next_node = self._mcts.UCTSEARCH(self._current_node)
        return best_next_node.state.get_action_card()

    def watch(self, observation, info):
        if info['done'] is True:
            pass
        elif info['is_new_round'] is True:
            self._current_node = None
        elif len(observation['playing_cards']) == 4:
            self._current_node = self._current_node.move_to_child(HeartState(observation, info))

