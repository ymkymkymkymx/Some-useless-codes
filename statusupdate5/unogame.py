import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils.utils import *

class RlUno(object):
    def __init__(self, playernum=2,human=0):
        self.env= rlcard.make('uno')
        self.player_num=playernum
        self.env.player_num=playernum
        self.env.game.num_players=playernum
        self.human=human
        self.action_num = self.env.game.get_action_num()
        self.state_shape = [7, 4, 15]
        self.timestep = self.env.timestep
        self.agents=[]
    def init_game(self):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        '''
        state, player_id = self.env.game.init_game()
        self.timestep = self.env.timestep
        return self.extract_state(state), player_id

    def step(self, action):
        
        self.timestep = self.env.timestep
        return self.env.step(action)

    def single_agent_step(self, action):
        
        self.timestep = self.env.timestep
        return self.env.single_agent_step( action)

    def reset(self):
        
        self.timestep = self.env.timestep
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
       
        self.agents=agents
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

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.init_game()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.env.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs        

       

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
        self.player_num=num
        self.env.player_num=num
        self.env.game.num_players=num
        return self.init_game()
    def get_player_vision(self,id):
        '''
        return the things that the player in game should be able to observes in a list of lists of strings
        infolist[0] stores the cards on the player's hand
        infolist[1] stores the 'target card'
        infolist[2] stores the number of cards on each players' hands
        infolist[3] stores the leagal actions
        Args:
            id: the player id you want to get vision of
        Return:
              infolist
        '''
        info=self.env.game.get_state(int(id))
        infolist=[]
        infolist.append(info['hand'])
        infolist.append(info['target'])
        cardleft=[]
        for i in range(self.player_num):
            cardleft.append(len(self.env.game.players[i].hand))
        infolist.append(cardleft)
        infolist.append(info['legal_actions'])
        return infolist

    def print_player_vision(self,id):
        '''
        print out things that the player in game should be able to observes in a list of lists of strings
        for debug purpose
        Args:
            id: the player id you want to get vision of    
        '''
        info=self.get_player_vision(id)
        print("Your card:")
        print(info[0])
        print("=======================================================================================================")
        print("Target card:")
        print(info[1])
        print("=======================================================================================================")
        print("Number of cards left: for each players:")
        for i in info[2]:
            print(i)
        print("=======================================================================================================")
        print("Leagal actions:")
        print(info[3])

    def askhumanforaction(self):
        '''
        This version is for debug and represent purposes, if you want to use it for webapp, pls replace the input() and printing with proper fucntions
        return: human action
        '''
        id=self.get_player_id()
        print("player {0}, now is your time to shine! Here are your states:".format(id))
        self.print_player_vision(id)
        info=self.get_player_vision(id)
        while True:
            print("Please make your move:")
            action=input()
            if action not in info[3] and action!='draw':
                print("Move not valid, please enter a valid move,\'draw\' if no moves can be make.")
            else:
                return action
    
    def gamewithai(self, humanorai,seed=None):
        '''
        gamewithai should start a game allows player to play against ai
        Args:
             seed: random's seed if provided for agents
             humanorai: a list of size player_num, humanorai[i]==-1 means the i-1 th player is human, huamnorai[i]==a where a >=0 means the i-1 th player is self.env.agents[a]
        '''
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
        #trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.init_game()
        state=self.env.game.get_state(0)
        # Loop to play the game
        #trajectories[player_id].append(state)
        
        while not self.is_over():
            #human plays
            next_state = state 
            next_player_id = player_id            
            if humanorai[player_id]==-1:
                action=self.askhumanforaction()
                next_state, next_player_id =self.env.game.step(action)
             # Agent plays
            else:     
                action = self.env.agents[humanorai[player_id]].eval_step(state)
                if action== None:
                    action='draw'
            # Environment steps
                next_state, next_player_id = self.env.game.step(action)
                print("Player {0} takes action: {1}".format(player_id,action))
            if self.is_over():
                print("Player {0} wins".format(player_id))
            # Set the state and player
            state = next_state
            player_id = next_player_id
    
    
    
    
if __name__ == '__main__':
    env=RlUno(4)
    agents=[]
    for i in range(env.player_num): 
        agents.append( RandomAgent(action_num=env.action_num))
    env.set_agents(agents)    
    env.init_game()
    env.print_state(0)
    env.print_player_vision(0)
    # 4 random agents example
    print("4 random agents example\n")
    ishuman=[0,1,2,3]
    env.gamewithai(ishuman)
    # 1 human and 3 random agents example
    print("\n 1 human and 3 random agents example \n")
    agents=[]
    for i in range(env.player_num-1): 
        agents.append( RandomAgent(action_num=env.action_num))   
    env.set_agents(agents)   
    ishuman=[-1,0,1,2]
    env.gamewithai(ishuman)