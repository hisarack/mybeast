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
        playing_cards = next_observation['playing_cards']
        playing_ids = next_observation['playing_ids']

        # generate my player card from valid hand cards
        valid_hand_cards = next_observation['hand_cards']
        if len(playing_cards) > 0:
            trick_suit = Card.get_suit_int(playing_cards[0])
            valid_hand_cards = [c for c in next_observation['hand_cards'] if Card.get_suit_int(c) == trick_suit]
        my_playing_card = random.choice(valid_hand_cards)
        next_observation['hand_cards'].remove(my_playing_card)
        playing_cards.append(my_playing_card)
        playing_ids.append(my_player_id)
        
        # generate competitor playing cards
        number_of_pending_players = number_of_players - len(playing_cards)
        for c in random.sample(next_info['available_cards'], number_of_pending_players):
            playing_cards.append(c)
            next_info['available_cards'].remove(c)
        for p in range(playing_ids[-1] + 1, playing_ids[-1] + number_of_pending_players + 1):
            playing_ids.append(p % number_of_players)
        
        print(playing_ids)
        print(playing_cards)
        print(len(next_info['available_cards']))
        print(len(next_observation['hand_cards']))

        # update observation and info
        next_observation['trick'] += 1
        looser_score, looser_player_id = self._evaluator.evaluate(playing_cards, playing_ids)
        if looser_player_id == my_player_id:
            next_observation['scores'][my_player_id] += looser_score
            next_observation['playing_cards'] = []
            next_observation['playing_ids'] = []
        else:
            end_player_id = my_player_id
            if looser_player_id > my_player_id:
                end_player_id += number_of_players
            next_playing_ids = [p % number_of_players for p in range(looser_player_id, end_player_id)]
            next_playing_cards = random.sample(next_info['available_cards'], len(next_playing_ids))
            for c in next_playing_cards:
                next_info['available_cards'].remove(c)
            next_observation['playing_ids'] = next_playing_ids
            next_observation['playing_cards'] = next_playing_cards
        
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

