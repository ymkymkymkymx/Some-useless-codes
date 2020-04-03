import tensorflow as tf

import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
from unogame import *
# Make environment
env =RlUno(2)
eval_env=RlUno(2)
# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 10000
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
##set_global_seed(0)
if True:
    sess1= tf.compat.v1.Session()

    # Set agents
    global_step = tf.Variable(0, name='global_step', trainable=False)
    agent = DQNAgent(sess1,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_init_size=memory_init_size,
                     norm_step=norm_step,
                     state_shape=env.state_shape,
                     mlp_layers=[100,100])
      
    

    sess1.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess1,root_path+'models/modelvrandom1.ckpt')
    
    sess2=tf.compat.v1.Session()
    agent2 = DQNAgent(sess2,
                     scope='dqn2',
                     action_num=env.action_num,
                     replay_memory_init_size=memory_init_size,
                     norm_step=norm_step,
                     state_shape=env.state_shape,
                     mlp_layers=[100,100])
      
    
    sess2.run(tf.global_variables_initializer())
    
    
    saver2=tf.train.Saver()
    saver2.restore(sess2,root_path+'models/model22.ckpt')
      
    eval_env.set_agents([agent,agent2])    

    if True:
        # Evaluate the performance
        if True:
            reward = 0
            rewardlist=[]
            for eval_episode in range(evaluate_num):
                _, payoffs = eval_env.run(is_training=False)
                reward += payoffs[0]
                rewardlist.append(payoffs[0])
            print('\n########## Evaluation ##########')
            print('Average reward is {}'.format(env.timestep, float(reward)/evaluate_num))
            print(rewardlist)
            print(rewardlist.count(1))