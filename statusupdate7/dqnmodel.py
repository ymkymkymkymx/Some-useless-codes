import tensorflow as tf

import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger
# Make environment


# Set the the number of steps for collecting normalization statistics
# and intial memory size
memory_init_size = 1000
norm_step = 100

# The paths for saving the logs and learning curves


## Set a global seed
##set_global_seed(0)
class DqnModel:
    def __init__(self):
        env=rlcard.make('uno')
        self.sess1= tf.compat.v1.Session()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.agent = DQNAgent(self.sess1,
                         scope='dqn',
                         action_num=env.action_num,
                         replay_memory_init_size=memory_init_size,
                         norm_step=norm_step,
                         state_shape=env.state_shape,
                         mlp_layers=[100,100])
        self.sess1.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver()
        self.saver.restore(self.sess1,'./experiments/uno_dqn_result/models/model1.ckpt')
      
       