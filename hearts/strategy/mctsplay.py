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



class MyHeartState(IState):

    def __init__(self, level, observation, info):
        self._level = level
        self._trick = observation['trick']
        self._player_id = info['my_player_id']
        self._hand_cards = info['my_hand_cards']
        self._playing_ids = observation['playing_ids']
        self._playing_cards = observation['playing_cards']
        self._number_of_players = observation['number_of_players']
        self._previous_action_card = info['previous_action_card']
        self._previous_player_id = info['previous_player_id']
        self._evaluator = Evaluator()
        self._info = info
        self._scores = observation['scores']
        self.num_moves = len(self._valid_hand_cards)

    def _get_valid_hand_cards(self, playing_cards, hand_cards):
        valid_hand_cards = hand_cards
        if len(playing_cards) > 0:
            trick_suit = Card.get_suit_int(playing_cards[0])
            valid_hand_cards = [c for c in hand_cards if Card.get_suit_int(c) == trick_suit]
        if len(valid_hand_cards) == 0:
            valid_hand_cards = hand_cards
        return valid_hand_cards

    def next_state(self):
        valid_hand_cards = self._get_valid_hand_cards(self._playing_cards, self,_hand_cards)
        my_playing_card = random.choice(valid_hand_cards)
        self._playing_ids.append(self._player_id)
        self._playing_cards.append(my_playing_card)
        
        next_observation = {}
        next_observation['number_of_players'] = self._number_of_players
        next_observation['scores'] = self._scores.copy()
        next_observation['hand_cards'] = self._hand_cards.copy()
        next_observation['hand_cards'].remove(my_playing_card)
        if len(self._playing_ids) == 4:
            looser_score, looser_player_id = self._evaluator.evaluate(self._playing_cards, self._playing_ids)
            next_observation['current_player_id'] = looser_player_id
            next_observation['playing_ids'] = []
            next_observation['playing_cards'] = []
            next_observation['trick'] = self._trick + 1
            if looser_player_id == self._player_id:
                next_observation['scores'][self._player_id] += looser_score
        else:
            next_observation['current_player_id'] = (self._player_id + 1) % self._number_of_players
            next_observation['playing_ids'] = self._playing_ids.copy()
            next_observation['playing_cards'] = self._playing_cards.copy()
            next_observation['trick'] = self._trick
        next_info = copy.depcopy(self._info)
        next_info['previous_action_card'] = my_playing_card
        next_info['previous_player_id'] = self._player_id
        return PlayerHeartState(self._level + 1, next_observation, next_info)

    def get_action_card(self):
        return self._previous_action_card

    def reward(self):
        r = 1.0 - self._scores[self._player_id] / 26
        return r

    def __hash__(self):
        hash_str = "{}:{}:{}:{}:{}".format(
            str(self._level),
            str(self._player_id),
            str(self._valid_hand_cards),
            str(self._playing_cards),
            str(self._playing_ids),
        ).encode('utf-8')
        return int(hashlib.md5(hash_str).hexdigest(), 16)

    def __eq__(self,other):
        if hash(self) == hash(other):
            return True
        return False
    



class PlayerHeartState(IState):

    def __init__(self, level, observation, info):
        self._level = level + 1
        self._trick = observation['trick']
        self._player_id = observation['current_player_id']
        self._available_cards = info['available_cards']
        self._playing_ids = observation['playing_ids']
        self._playing_cards = observation['playing_cards']
        self._previous_action_card = info['previous_action_card']
        self._previous_player_id = info['previous_player_id']
        self._evaluator = Evaluator()
        self._scores = observation['scores']

    def next_state(self):
        competitor_playing_card = random.choice(self._available_cards)
        self._playing_ids.append(self._player_id)
        self._playing_cards.append(my_playing_card)
        
        next_observation = {}
        next_observation['number_of_players'] = self._number_of_players
        next_observation['scores'] = self._scores.copy()
        if len(self._playing_ids) == 4:
            looser_score, looser_player_id = self._evaluator.evaluate(self._playing_cards, self._playing_ids)
            next_observation['current_player_id'] = looser_player_id
            next_observation['playing_ids'] = []
            next_observation['playing_cards'] = []
            next_observation['trick'] = self._trick + 1
            if looser_player_id == self._player_id:
                next_observation['scores'][self._player_id] += looser_score
        else:
            next_observation['current_player_id'] = (self._player_id + 1) % self._number_of_players
            next_observation['playing_ids'] = self._playing_ids.copy()
            next_observation['playing_cards'] = self._playing_cards.copy()
            next_observation['trick'] = self._trick
        next_info = copy.depcopy(self._info)
        next_info['previous_action_card'] = competitor_playing_card
        next_info['previous_player_id'] = self._player_id
        next_info['available_cards'].remove(my_playing_card)
        if next_observation['current_player_id'] == next_info['my_player_id']:
            return MyHeartState(self._level + 1, next_observation, next_info)
        else:
            return PlayerHeartState(self._level + 1, next_observation, next_info)

    def terminal(self):
        return self._level >= 53
    
    def reward(self):
        r = 1.0 - self._scores[next_info['my_player_id']] / 26
        return r
 
    def __hash__(self):
        hash_str = "{}:{}:{}:{}:{}".format(
            str(self._level),
            str(self._player_id),
            str(self._available_cards),
            str(self._playing_cards),
            str(self._playing_ids),
        ).encode('utf-8')
        return int(hashlib.md5(hash_str).hexdigest(), 16)
    

    def __eq__(self,other):
        if hash(self) == hash(other):
            return True
        return False
    

class MCTSPlayStrategy(strategy.IStrategy):
    
    def __init__(self, budget, my_player_id):
        self._my_player_id = my_player_id
        self._evaluator = Evaluator()
        self._mcts = MCTS(budget)
        self._current_node = None

    def move(self, observation):
        if self._current_node is None:
            new_mcts_info = {}
            new_mcts_info['my_player_id'] = self._my_player_id
            new_mcts_info['my_hand_cards'] = observation['hand_cards'].copy()
            deck = Deck()
            new_mcts_info['available_cards'] = deck.draw(52)
            for c in new_mcts_info['my_hand_cards']:
                new_mcts_info['available_cards'].remove(c)
            self._current_node = Node(MyHeartState(0, observation, new_mcts_info))
            if len(observation['playing_ids']) == 0:
                return Card.new('2c')
        best_next_node = self._mcts.UCTSEARCH(self._current_node, observation)
        return best_next_node.state.get_action_card()

    def _show_cards(self, hand_cards):
        suitrank_ints = [c & 0xFF00 for c in hand_cards] 
        sorted_indices = sorted(range(len(suitrank_ints)), key=lambda k: suitrank_ints[k])
        sorted_cards = [hand_cards[ind] for ind in sorted_indices]
        qq = [Card.int_to_pretty_str(c) for c in sorted_cards]
        print(' '.join(qq))

    def watch(self, observation, info):
        if info['done'] is True:
            pass
        elif info['is_new_round'] is True:
            self._current_node = None
        elif info['current_player_id'] == self._my_player_id:
            my_played_card = observation['playing_cards'][observation['playing_ids'].index(self._my_player_id)]
            my_played_reward = 0
            if info['punish_player_id'] == self._my_player_id:
                my_played_reward = info['punish_score']
            current_mcts_info = self._current_node.state.get_mcts_info()
            new_available_cards = current_mcts_info['available_cards'].copy()
            for c in observation['playing_cards']:
                if c in new_available_cards:
                    new_available_cards.remove(c)
            new_mcts_info['available_cards'] = new_available_cards
            new_mcts_info['my_player_id'] = self._my_player_id
            new_mcts_info['my_played_card'] = my_played_card
            new_mcts_info['my_played_reward'] = my_played_reward
            new_mcts_info['my_hand_cards'] = current_mcts_info['my_hand_cards'].copy()
            if my_played_card in new_mcts_info['my_hand_cards']:
                new_mcts_info['my_hand_cards'].remove(my_played_card)
            new_mcts_info['punish_player_id'] = info['punish_player_id']
            next_state = HeartState(observation, new_mcts_info)
            self._current_node = self._current_node.move_to_child(next_state)


