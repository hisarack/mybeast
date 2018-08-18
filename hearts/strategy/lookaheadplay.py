import gym
from gymhearts import env as hearts_env
from gymhearts import strategy
from gymhearts.evaluator import Evaluator

from treys import Deck
from treys import Card

from .completeplay import CompletePlayStrategy


# Focus on do not loose current trick when we are last player
# and play smallest card if we are not last player.
# optimize the rule if we can not win current trick or do not get punish score, drop the worst card
class LookAheadPlayStrategy(strategy.IStrategy):

    def __init__(self):
        self._evaluator = Evaluator()
        deck = Deck()
        self._available_cards = deck.draw(52)

    def move(self, observation):
        number_of_playing_ids = len(observation['playing_ids'])
        valid_hand_cards = observation['valid_hand_cards']
        playing_cards = observation['playing_cards']
        valid_hand_ranks = [Card.get_rank_int(c) for c in valid_hand_cards]
        min_card_id = valid_hand_ranks.index(min(valid_hand_ranks))
        max_card_id = valid_hand_ranks.index(max(valid_hand_ranks))

        # if i am last player in this trick
        if number_of_playing_ids == observation['number_of_players']-1:
            first_suit = Card.get_suit_int(playing_cards[0])
            competitor_ranks = [Card.get_rank_int(c) for c in playing_cards if Card.get_suit_int(c) == first_suit]
            max_suit = Card.get_suit_int(valid_hand_cards[max_card_id])
            punish_score = self._evaluator.calculate_score(playing_cards)
            # we do not have same suit card, so just drop max card. it does not get punish score
            if (max_suit != first_suit) or (punish_score == 0):
                return valid_hand_cards[max_card_id]
            # we will be the looser in this round, so just drop max card.
            elif valid_hand_ranks[min_card_id] > min(competitor_ranks):
                return valid_hand_cards[max_card_id]
            # drop the card which is not larger then biggest playing card.
            else:
                max_safe_card_id = min_card_id
                max_safe_rank = valid_hand_ranks[min_card_id]
                max_competitor_ranks = max(competitor_ranks)
                for tmp_card_id, r in enumerate(valid_hand_ranks):
                    if (r < max_competitor_ranks) and (r > max_safe_rank):
                        max_safe_rank = r
                        max_safe_card_id = tmp_card_id
                return valid_hand_cards[max_safe_card_id]
        # apply monte carlo sampling to estimate win rate of valid hand cards
        else:
            simulated_scores = [self._simulate(card, observation) for card in valid_hand_cards]
            best_card_id = simulated_scores.index(min(simulated_scores))
            return valid_hand_cards[best_card_id]

    def _simulate(self, expanded_card, observation):
        competitor_cards = self._available_cards.copy()
        hand_cards = observation['hand_cards'].copy()
        for c in hand_cards:
            competitor_cards.remove(c)
        # play current trick
        
        # play util finish the round and use complete strategy as default policy
        Deck._FULL_DECK = competitor_cards
        deck = Deck()
        Deck._FULL_DECK = []
        env = hearts_env.HeartsEnv()
        env.add_player(
            hand_cards=hand_cards, 
            strategy=CompletePlayStrategy()
        )
        for i in range(1, observation['number_of_players']):
            env.add_player(
                hand_cards=deck.draw(len(hand_cards)), 
                strategy=CompletePlayStrategy()
            )
        observation = env.get_observation()
        while not observation['is_new_round']:
            action = env.move()
            observation, reward, done, info = env.step(action)
        return observation['scores'][0]

    def watch(self, observation, info):
        if info['done'] is True:
            pass
        elif info['is_new_round'] is True:
            deck = Deck()
            self._available_cards = deck.draw(52)
        else:
            played_card = info['action']
            self._available_cards.remove(played_card)
        

