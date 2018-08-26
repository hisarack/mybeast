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
        self._observation = copy.deepcopy(observation)
        self._info = copy.deepcopy(info)
        self._evaluator = Evaluator()
        
        # calculate number of possible combination
        number_of_available_cards = len(info['available_cards'])
        number_of_pending_players = observation['number_of_players'] - len(observation['playing_cards']) - 1
        self.qq = []
        self.num_moves = 1
        for n in range(number_of_available_cards, number_of_available_cards-number_of_pending_players, -1):
            self.num_moves *= n
            self.qq.append(n)
        valid_hand_cards = self._get_valid_hand_cards(
            observation['playing_cards'], 
            info['my_hand_cards']
        )
        self.num_moves *= len(valid_hand_cards)
        self.qq.append(len(valid_hand_cards))

    def _get_valid_hand_cards(self, playing_cards, hand_cards):
        valid_hand_cards = hand_cards
        if len(playing_cards) > 0:
            trick_suit = Card.get_suit_int(playing_cards[0])
            valid_hand_cards = [c for c in hand_cards if Card.get_suit_int(c) == trick_suit]
            if len(valid_hand_cards) == 0:
                valid_hand_cards = hand_cards
        return valid_hand_cards

    def _create_next_observation(self):
        # prepare the playing cards and ids
        number_of_players = self._observation['number_of_players']
        first_player_id = 0
        if 'punish_player_id' in self._info:
            first_player_id = self._info['punish_player_id']
        my_player_id = self._info['my_player_id']
        end_player_id = my_player_id
        if first_player_id > my_player_id:
            end_player_id += number_of_players
        playing_ids = [p % number_of_players for p in range(first_player_id, end_player_id)]
        playing_cards = random.sample(self._info['available_cards'], len(playing_ids))
        next_observation = copy.deepcopy(self._observation)
        next_observation['playing_ids'] = playing_ids
        next_observation['playing_cards'] = playing_cards
        return next_observation

    def next_state(self, observation=None):
        if observation is None:
            observation = self._create_next_observation()
        next_observation = copy.deepcopy(observation)
        next_info = copy.deepcopy(self._info)
        trick = next_observation['trick']
        my_player_id = next_info['my_player_id']
        number_of_players = next_observation['number_of_players']
        playing_cards = next_observation['playing_cards']
        playing_ids = next_observation['playing_ids']

        # generate my player card from valid hand cards
        valid_hand_cards = self._get_valid_hand_cards(playing_cards, next_info['my_hand_cards'])
        my_playing_card = random.choice(valid_hand_cards)
        next_info['my_hand_cards'].remove(my_playing_card)
        playing_cards.append(my_playing_card)
        playing_ids.append(my_player_id)
        
        # generate competitor playing cards and remove them from available cards
        for c in playing_cards:
            if c in next_info['available_cards']:
                next_info['available_cards'].remove(c)
        if len(playing_cards) < number_of_players:
            number_of_pending_players = number_of_players - len(playing_cards)
            for c in random.sample(next_info['available_cards'], number_of_pending_players):
                playing_cards.append(c)
                next_info['available_cards'].remove(c)
            for p in range(playing_ids[-1] + 1, playing_ids[-1] + number_of_pending_players + 1):
                playing_ids.append(p % number_of_players)

        # update observation and info
        next_observation['trick'] += 1
        next_info['my_played_card'] = my_playing_card
        next_info['my_played_reward'] = 0
        looser_score, looser_player_id = self._evaluator.evaluate(playing_cards, playing_ids)
        next_info['punish_player_id'] = looser_player_id
        if looser_player_id == my_player_id:
            next_info['my_played_reward'] = looser_score
            next_observation['scores'][my_player_id] += looser_score
        next_observation['playing_ids'] = playing_ids
        next_observation['playing_cards'] = playing_cards

        return HeartState(next_observation, next_info)
    
    def update_mcts_info(self, info):
        self._info = copy.deepcopy(info)
    
    def get_mcts_info(self):
        return self._info

    def get_action_card(self):
        return self._info['my_played_card']

    def terminal(self):
        return len(self._info['available_cards']) == 0 & len(self._info['my_hand_cards']) == 0
    
    def reward(self):
        r = 1.0 - self._info['my_played_reward'] / 26
        return r
 
    def get_my_hand_cards(self):
        suitrank_ints = [c & 0xFF00 for c in self._info['my_hand_cards']]
        sorted_indices = sorted(range(len(suitrank_ints)), key=lambda k: suitrank_ints[k])
        sorted_cards = [self._info['my_hand_cards'][ind] for ind in sorted_indices]
        return sorted_cards

    def __hash__(self):
        hash_str = "{}:{}:{}:{}:{}".format(
            str(self._observation['trick']),
            str(self._info['available_cards']),
            str(self._info['my_hand_cards']),
            str(self._observation['playing_cards']),
            str(self._observation['playing_ids']),
        ).encode('utf-8')
        return int(hashlib.md5(hash_str).hexdigest(), 16)
    

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
            new_mcts_info = {}
            new_mcts_info['my_player_id'] = self._my_player_id
            new_mcts_info['my_hand_cards'] = observation['hand_cards'].copy()
            deck = Deck()
            new_mcts_info['available_cards'] = deck.draw(52)
            for c in new_mcts_info['my_hand_cards']:
                new_mcts_info['available_cards'].remove(c)
            self._current_node = Node(HeartState(observation, new_mcts_info))
        best_next_node = self._mcts.UCTSEARCH(self._current_node, observation)
        return best_next_node.state.get_action_card()

    def watch(self, observation, info):
        if info['done'] is True:
            pass
        elif info['is_new_round'] is True:
            self._current_node = None
            print('new round')
        elif len(observation['playing_cards']) == 4:
            my_played_card = observation['playing_cards'][observation['playing_ids'].index(self._my_player_id)]
            my_played_reward = 0
            if info['punish_player_id'] == self._my_player_id:
                my_played_reward = info['punish_score']
            current_mcts_info = self._current_node.state.get_mcts_info()
            new_available_cards = current_mcts_info['available_cards'].copy()
            for c in observation['playing_cards']:
                if c in new_available_cards:
                    new_available_cards.remove(c)
            new_mcts_info = {}
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
