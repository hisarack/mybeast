from pycfr.pokertrees import *
from pycfr.pokergames import *
from pycfr.card import Card

players = 2
deck = [Card(14,1),Card(13,2),Card(13,1),Card(12,1)]
rounds = [
    RoundInfo(holecards=1,boardcards=0,betsize=2,maxbets=[2,2]),
    RoundInfo(holecards=0,boardcards=1,betsize=4,maxbets=[2,2])]

ante = 1
blinds = [1,2]
gamerules = GameRules(players, deck, rounds, ante, blinds, handeval=leduc_eval)
gametree = GameTree(gamerules)
gametree.build()
