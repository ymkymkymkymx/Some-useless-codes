import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed

class RlUno(object):
    def __init__(self, playernum=2,human=0):
        self.env= rlcard.make('uno')
        self.env.player_num=playernum
        self.env.game.num_players=playernum
        self.human=human
        
    def init_game(self):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        '''
        state, player_id = self.env.game.init_game()
        return self.extract_state(state), player_id

    def step(self, action):
        

        return self.env.step(action)

    def single_agent_step(self, action):
        

        return self.env.single_agent_step( action)

    def reset(self):
        

        return self.env.extract_state(state)

    def step_back(self):
        

        return self.env.step_back

    def get_player_id(self):
        ''' Get the current player id

        Returns:
            (int): the id of the current player
        '''
        return self.env.game.get_player_id()

    def is_over(self):
        ''' Check whether the curent game is over

        Returns:
            (boolean): True is current game is over
        '''
        return self.env.game.is_over()

    def get_state(self, player_id):
        ''' Get the state given player id

        Args:
            player_id (int): The player id

        Returns:
            (numpy.array): The observed state of the player
        '''
        return self.env.extract_state(self.env.game.get_state(player_id))

    def set_agents(self, agents):
        ''' Set the agents that will interact with the environment

        Args:
            agents (list): List of Agent classes
        '''
       

        self.env.set_agents( agents)

    def run(self, is_training=False, seed=None):
        ''' Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.
            seed (int): A seed for running the game. For single-process program,
              the seed should be set to None. For multi-process program, the
              seed should be asigned for reproducibility.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        

        return self.env.run(is_training,seed)

    def run_multi(self, task_num, result, is_training=False, seed=None):
        self.env.run_multi(task_num, result, is_training, seed)

    def set_mode(self, active_player=0, single_agent_mode=False, human_mode=False):
        ''' Turn on the single-agent-mode. Pretrained models will
            be loaded to simulate other agents

        Args:
            active_player (int): The player that does not use pretrained models
        '''
        self.env.set_mode(active_player, single_agent_mode, human_mode)

    def print_state(self, player):
        ''' Print out the state of a given player

        Args:
            player (int): Player id
        '''
        self.env.print_state(player)

    def print_result(self, player):
        ''' Print the game result when the game is over

        Args:
            player (int): The human player id
        '''
        self.env.print_result(player)

    @staticmethod
    def print_action(action):
        ''' Print out an action in a nice form

        Args:
            action (str): A string a action
        '''
        self.env.print_action(action)

    def load_model(self):
        ''' Load pretrained/rule model

        Returns:
            model (Model): A Model object
        '''
        return self.env.load_model()

    def extract_state(self, state):
        ''' Extract useful information from state for RL. Must be implemented in the child class.

        Args:
            state (dict): the raw state

        Returns:
            (numpy.array): the extracted state
        '''
        return self.env.extract_state(state)

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            (list): A list of payoffs for each player.

        Note: Must be implemented in the child class.
        '''
        return self.env.get_payoffs()

    def decode_action(self, action_id):
        ''' Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (string): The action that will be passed to the game engine.

        Note: Must be implemented in the child class.
        '''
        return self.env.decode_action(action_id)

    def get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (list): A list of legal actions' id.

        Note: Must be implemented in the child class.
        '''
        return self.env.get_legal_actions()
    
    def set_player_number(self,num):
        '''
        Set the player number for Uno game, and reinitialize the game
        Args: 
             num: the number of player
        '''
        self.env.player_num=num
        self.env.game.num_players=num
        return self.init_game()
    
if __name__ == '__main__':
    env=RlUno(4)