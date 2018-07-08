import gym
import gymholdem

env = gymholdem.TexasHoldemEnv(2)
env.add_player(0, stack=2000)
env.add_player(1, stack=2000)

(player_states, (community_infos, community_cards)) = env.reset()
(player_infos, player_hands) = zip(*player_states)

env.render(mode='human')

done = False
while not done:
    action = gymholdem.safe_actions(community_infos, n_seats=env.n_seats)
    (player_states, (community_infos, community_cards)), rews, done, info = env.step(action)
    env.render(mode='human')
