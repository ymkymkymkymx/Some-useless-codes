import tensorflow as tf

import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from unogame import *
# Make environment
env =RlUno(2)
eval_env = RlUno(2)
# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 1000
episode_num = 10000

# Set the the number of steps for collecting normalization statistics
# and intial memory size
memory_init_size = 1000
norm_step = 100

# The paths for saving the logs and learning curves
root_path = './experiments/uno_dqn_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'

## Set a global seed
## set_global_seed(0)
if True:
    sess1= tf.compat.v1.Session()
    sess2=tf.compat.v1.Session()
    # Set agents
    global_step = tf.Variable(0, name='global_step', trainable=False)
    agent = DQNAgent(sess1,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_init_size=memory_init_size,
                     norm_step=norm_step,
                     state_shape=env.state_shape,
                     mlp_layers=[100,100])
    agent2 = DQNAgent(sess2,
                     scope='dqn2',
                     action_num=env.action_num,
                     replay_memory_init_size=memory_init_size,
                     norm_step=norm_step,
                     state_shape=env.state_shape,
                     mlp_layers=[100,100])
      
    env.set_agents([agent,agent2])
    eval_env.set_agents([agent,agent2])

    sess1.run(tf.global_variables_initializer())
    sess2.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess1,root_path+'models/model11.ckpt')
    saver.restore(sess2,root_path+'models/model22.ckpt')      
    # Count the number of steps
    step_counter = 0
    step_counter2 = 0
    # Init a Logger to plot the learning curve
    logger = Logger(xlabel='timestep', ylabel='reward', legend='DQN against DQN on UNO', log_path=log_path, csv_path=csv_path)

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train
        for ts in trajectories[0]:
            agent.feed(ts)
            step_counter += 1

            # Train the agent
            if step_counter > memory_init_size + norm_step:
                loss = agent.train()
                #print('\rINFO - Step {}, loss: {}'.format(step_counter, loss), end='')
        # Feed transitions into agent memory, and train
        for ts in trajectories[1]:
            agent2.feed(ts)
            step_counter2 += 1

            # Train the agent
            if step_counter2 > memory_init_size + norm_step:
                loss = agent2.train()
                #print('\rINFO - Step {}, loss: {}'.format(step_counter, loss), end='')
        # Evaluate the performance
        if episode % evaluate_every == 0:
            reward = 0
            rewardlist=[]
            for eval_episode in range(evaluate_num):
                _, payoffs = eval_env.run(is_training=False)
                reward += payoffs[0]
                rewardlist.append(payoffs[0])
            logger.log('\n########## Evaluation ##########')
            logger.log('Timestep: {} Average reward is {}'.format(env.timestep, float(reward)/evaluate_num))
            print(rewardlist)
            # Add point to logger
            logger.add_point(x=env.timestep, y=float(reward)/evaluate_num)

        # Make plot
        if episode % save_plot_every == 0 and episode > 0:
            logger.make_plot(save_path=figure_path+str(episode)+'.png')
    saver.save(sess1,save_path=root_path+'models/model11.ckpt')
    saver.save(sess2,save_path=root_path+'models/model22.ckpt')
    # Make the final plot
    logger.make_plot(save_path=figure_path+'final_'+str(episode)+'.png')