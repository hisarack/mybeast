from pycfr.pokertrees import *
from pycfr.pokergames import *
from pycfr.card import Card
from pycfr.pokercfr import CounterfactualRegretMinimizer as CFR

hskuhn = half_street_kuhn_rules()
cfr = CFR(hskuhn)

iters_per_block = 1000
blocks = 10
for block in xrange(blocks):
    print("Iteration {}".format(block * iters_per_block))
    cfr.run(iters_per_block)
    result = cfr.profile.best_response()
    print("Best response EV: {}".format(result[1]))
    print("Total exploitability: {}".format(sum(result[1])))

# FOLD = 0
# CALL = 1
# RAISE = 2
nash_strategies = cfr.profile
for strategy in nash_strategies.strategies:
    print(strategy.policy)

