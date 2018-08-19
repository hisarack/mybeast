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
    NUM_TURNS = 13   
    GOAL = 0
    MOVES=[2,-2,3,-3]
    MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
    num_moves=len(MOVES)

    def __init__(self, observation, info):
        deck = Deck()
        self._available_cards = []
        self._score = score
        self._trick = trick
        self._playing_cards = playing_cards
        self._player_id = player_id
        self._evaluator = Evaluator()
        self._done = done

    def next_state(self):
        trick = observation['trick']
        player_id = observation['current_player_id']
        number_of_players = observation['number_of_players']
        playing_cards = random.shuffle(info['competitor_cards'], number_of_players-1)
        playing_cards.insert(player_id, random.choice(observation['valid_hand_cards']))
        looser_score, looser_player_id = self._evaluator.evalute(playing_cards, playing_ids)
        next_score = self._score
        if looser_player_id != 0:
            next_score += looser_score
        done = False
        if len(info['competitor_cards']) == number_of_players-1:
            done = True
        return State(
            score = next_score,
            trick = self._trick + 1,
            playing_cards = playing_cards,
            player_id = player_id,
            done = done
        )
    
    def get_action_card(self):
        return self._playing_cards[self._player_id]

    def terminal(self):
        return self._done
    
    def reward(self):
        r = 1.0 - self._score / 26
        return r
    
    def __hash__(self):
        return int(hashlib.md5(
            (self._trick + ":" + str(self._played_cards)).encode('utf-8')
        ).hexdigest(), 16)
    
    def __eq__(self,other):
        if hash(self) == hash(other):
            return True
        return False
    
    def __repr__(self):
        return "Score: %d; Playing Cards: %s "% (self._score, self._played_cards)


class MCTSPlayStrategy(strategy.IStrategy):
    
    def __init__(self, budget):
        self._evaluator = Evaluator()
        deck = Deck()
        self._available_cards = deck.draw(52)
        self._mcts = MCTS(budget)
        self._current_node = Node(HeartState(
            score = 0,
            trick = 0,
            playing_cards = [],
            done = False
        ))

    def move(self, observation):
        competitor_cards = self._available_cards.copy()
        hand_cards = observation['hand_cards']
        for c in hand_cards:
            competitor_cards.remove(c)
        info = {}
        info['competitor_cards'] = competitor_cards
        best_next_node = self._mcts.UCTSEARCH(self._current_node)
        return best_next_node.state.get_action_card()

    def watch(self, observation, info):
        if info['done'] is True:
            pass
        elif info['is_new_round'] is True:
            deck = Deck()
            self._available_cards = deck.draw(52)
        else:
            played_card = info['action']
            self._available_cards.remove(played_card)
        playing_cards = observation['playing_cards']
        if len(playing_cards) == 4:
            self._current_node.move_to_child(HeartState(
                score = score,
                trick = observation['trick'],
                playing_cards = played_cards,
                done = info['done']
            ))

