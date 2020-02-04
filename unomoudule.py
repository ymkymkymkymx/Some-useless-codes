import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed

# Make environment
env = rlcard.make('uno')
episode_num = 2
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
        