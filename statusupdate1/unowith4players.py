import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed
'''
This is a modification of the example code on http://rlcard.org/getting_started.html.
The original code was rlcard's blackjack example
I modified it to play uno instead and adjusted the player number and output format.
Mark Yu
Feb  4 2020
'''
# Make environment
env = rlcard.make('uno')
episode_num = 2
# There is no build in function that can change the player numbers of a game so I checked and changed the status for all dependencies that relevent to the game status and initialization
env.player_num=4
env.game.num_players=4
env.init_game()
# Set a global seed
set_global_seed(0)
agents=[]
# Set up agents
for i in range(env.player_num): 
    agents.append( RandomAgent(action_num=env.action_num))
env.set_agents(agents)

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=False)

    # Print out the trajectories
    print('\nEpisode {}'.format(episode))
    for ts in trajectories[0]:
        print('Action: {}, Reward: {}, Done: {}'.format( ts[1], ts[2], ts[4]))
        