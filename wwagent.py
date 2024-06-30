

"""
Kayla Laufer
Artificial Intelligence Spring 2024

Overview:
This project focuses on developing a rational agent to effectively navigate and succeed in the Wumpus World game, 
a grid-based environment where the agent must avoid pitfalls and a monster called the Wumpus to collect gold. 
The challenge lies in making decisions with incomplete information and potential dangers lurking in unexplored 
areas. The provided simulation tools include wwsim.py for the game environment and an initial agent (original 
wwagent.py) programmed to make decisions randomly. This setup serves as a basis for enhancing decision-making 
capabilities through advanced artificial intelligence techniques.

Objective:
The primary goal is to engineer a rational agent capable of navigating the partially observable and perilous 
state space of the Wumpus World. The agent integrates planning with reinforcement learning to safely and efficiently 
achieve its objectives.

Approach:
The agent employs an epsilon-greedy policy, balancing between exploration and exploitation to optimize decision-making 
in uncertain environments. It utilizes probabilistic model checking for safe path planning, relying on truth-table 
enumeration to infer the safest moves based on available sensory data. Temporal difference reinforcement learning further 
refines its strategy over time, learning from past actions to enhance future performance.

This enhanced approach allows the agent to surpass the limitations of the original random decision-making strategy, 
employing both learned experiences and logical deductions to navigate the Wumpus World effectively.


Modified from wwagent.py written by Greg Scott

# FACING KEY:
#    0 = up
#    1 = right
#    2 = down
#    3 = left

# Actions
# 'move' 'grab' 'shoot' 'left' right'

"""
import copy
from random import randint
import numpy as np

# This is the class that represents an agent

class WWAgent:

    def __init__(self):
        self.max=4 # number of cells in one side of square world
        self.stopTheAgent=False # set to true to stop th agent at end of episode
        self.position = (0, 3) # top is (0,0)
        self.facing = 'right'
        self.arrow = 1
        self.percepts = (None, None, None, None, None)
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)] 
        self.visited = [] # List of rooms the agent has visited
        self.safe = [] # List of safe rooms -- no pit/wumpus
        self.alpha_count = 0 # Number of models where alpha is true
        self.model_count = 0 # Number of true models
        self.q_table = np.zeros((4, 4, 4, 4)) # Q-table for SARSA
        self.epsilon = 0.1 # Default epsilon for greedy-epsilon probability
        self.learning_rate = 0.5 # Default learning rate
        self.discount = 0.8 # Default discount
        self.give_up = False # Flag for no good moves (gaurentee to die) -- will end the game
        self.rotate = False # Flag for alphas that require multiple rotations
        self.prob_safe = {'03':1.0} # Dictionary for all alphas with their corresponding probability
        self.prev_alpha = '03' # Initialize to start position
        self.dead = False # Agent lost the game
        self.gold = False # Agent won the game
        self.num_buckets = 4 # Amount of probability buckets
        self.buckets = np.linspace(0, 1, self.num_buckets+1) # Initialize bucket ranges
        self.prev_state = [0, 3, 3, 2] # Default - x, y, prob bucket, action
        self.rl_count = 0 # Counts the amount of times SARSA is choosen for chose action
        self.mc_count = 0 # Counts the amount of times model checking is choosen for chose action
        self.directions = { # Dictionary of index to their direction value
            0 : 'up',
            1 : 'down',
            2 : 'left',
            3 : 'right'
        }
        print("New agent created")
        
    def reset(self):
        self.max=4 # number of cells in one side of square world
        self.stopTheAgent=False # set to true to stop th agent at end of episode
        self.position = (0, 3) # top is (0,0)
        self.facing = 'right'
        self.arrow = 1
        self.percepts = (None, None, None, None, None)
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)] 
        self.visited = [] # List of rooms the agent has visited
        self.safe = [] # List of safe rooms -- no pit/wumpus
        self.alpha_count = 0 # Number of models where alpha is true
        self.model_count = 0 # Number of true models
        self.q_table = np.zeros((4, 4, 4, 4)) # Q-table for SARSA
        self.epsilon = 0.1 # Default epsilon for greedy-epsilon probability
        self.learning_rate = 0.5 # Default learning rate
        self.discount = 0.8 # Default discount
        self.give_up = False # Flag for no good moves (gaurentee to die) -- will end the game
        self.rotate = False # Flag for alphas that require multiple rotations
        self.prob_safe = {'03':1.0} # Dictionary for all alphas with their corresponding probability
        self.prev_alpha = '03' # Initialize to start position
        self.dead = False # Agent lost the game
        self.gold = False # Agent won the game
        self.num_buckets = 4 # Amount of probability buckets
        self.buckets = np.linspace(0, 1, self.num_buckets+1) # Initialize bucket ranges
        self.prev_state = [0, 3, 3, 2] # Default - x, y, prob bucket, action
        self.rl_count = 0 # Counts the amount of times SARSA is choosen for chose action
        self.mc_count = 0 # Counts the amount of times model checking is choosen for chose action
        self.directions = { # Dictionary of index to their direction value
            0 : 'up',
            1 : 'down',
            2 : 'left',
            3 : 'right'
        }
        print("New agent created")

    """
    Calculate the new position on a 2D grid based on a specified movement direction.

    Parameters:
    - current_position (tuple): A tuple (x, y) indicating the current coordinates.
    - move_command (str): A command as a string indicating the direction to move.
      Valid commands are 'up', 'down', 'left', 'right'.
    
    Returns:
    - str: A string representation of the new position after the move, concatenating
      the x and y coordinates without any separator (e.g., 'x''y').
      
    This function does ensures that the new position does not go out of bounds. The given 
    input has also been checked. 
    """
    def move_position(self, current_position, move_command):
        # Movement effects for each command
        # Check if coordinates are within the bounds of the grid
        if not (0 <= current_position[0] < 4 and 0 <= current_position[1] < 4):
            raise ValueError("x and y must be within the grid boundaries (0 to 3).")
    
        movements = {
            'up': (0, -1),  # Move up decreases y
            'down': (0, 1),  # Move down increases y
            'left': (-1, 0),  # Move left decreases x
            'right': (1, 0)  # Move right increases x
        }

        # Get the delta from the movements dictionary
        delta_x, delta_y = movements[move_command]

        # Apply the delta to the current position
        new_x = current_position[0] + delta_x
        new_y = current_position[1] + delta_y

        if not (0 <= new_x < 4 and 0 <= new_y < 4):
            raise ValueError("x and y must be within the grid boundaries (0 to 3).")

        # Ensure the new position does not go out of the grid boundaries
        """new_x = max(0, min(new_x, self.max - 1))
        new_y = max(0, min(new_y, self.max - 1))"""

        # Return the new position as a string in the 'xy' format
        return str(new_x) + str(new_y)

    """
    Set the parameters for the learning algorithm.

    This function configures the agent with key parameters that affect its learning
    and decision-making process within a reinforcement learning environment.

    Parameters:
    - epsilon (float): The exploration rate. It determines the probability of choosing
      a random action over the best action according to the Q-table. Ranging from 0-1
    - learning_rate (float): The learning rate or step size. It determines how much
      new information overrides old information. Ranging from 0-1
    - discount (float): The discount factor used in the update rule. It quantifies how
      much importance is given to future rewards. A factor of 0 will make the agent
      short-sighted by considering only current rewards, while a factor close to 1
      will make it strive for long-term high rewards.

    Each of these parameters plays a crucial role in the convergence and performance
    of the learning algorithm. Improper settings can lead to suboptimal learning
    behavior or failure to converge to a solution.
    """
    def set_param(self, epsilon, learning_rate, discount):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
    
    """
    set_table does a deep copy of the q-table given by the simulation
    Param:
        - table (four dimensional array): q-table given by simulation
    """
    def set_table(self, table):
        self.q_table = copy.deepcopy(table)
       
    """
    This sets the gold field to true if called.
    """  
    def has_gold(self):
        self.gold = True
    """
    This sets the dead field to true if called.
    """
    def is_dead(self):
        self.dead = True

    # Add the latest percepts to list of percepts received so far
    # This function is called by the wumpus simulation and will
    # update the sensory data. The sensor data is placed into a
    # map structured KB for later use
    # It also updates the visited and safe list
    
    def update(self, percept):
        self.percepts=percept
        
        #[stench, breeze, glitter, bump, scream]
        if self.position[0] in range(self.max) and self.position[1] in range(self.max):
            # puts the percept at the spot in the map where sensed
            self.map[ self.position[0]][self.position[1]]=self.percepts
           
            # Knows which rooms it has visited, hence they are OK
            visited_loc = str(self.position[0]) + str(self.position[1])
            if not visited_loc in self.visited or self.visited.count(visited_loc) < 5:
                self.visited.append(visited_loc)
            if not visited_loc in self.safe:
                self.safe.append(visited_loc)
                # if there are no senses in current loc then adj rooms are safe to move to
                if percept[0] is None and percept[1] is None:
                    adj = self.get_adjacent_indices(int(visited_loc[0]), int(visited_loc[1]))
                    for room in adj:
                        if room not in self.safe:
                            self.safe.append(room)
            else: # Visited is in list, but are their safe adj rooms in the list?
                if percept[0] is None and percept[1] is None:
                    adj = self.get_adjacent_indices(int(visited_loc[0]), int(visited_loc[1]))
                    for room in adj:
                        if room not in self.safe:
                            self.safe.append(room)
        print('SAFE: ', self.safe)
        

    """ This function takes the room type and position on the grid as arguments. 
    It calculates the adjacent indicies for that location. 
    Param:
        - type (string): type of room
        - x (int): grid row of current position
        - y (int): grid column of current position
    Return:
        - a list of adjacent rooms
    """
    def get_adjacent_rooms(self, type, x, y):
        if not (0 <= x < 4 and 0 <= y < 4):
            raise ValueError("x and y must be within the grid boundaries (0 to 3).")
        
        max_x = 4 
        max_y = 4
        adjacent_indices = []
        # Check each direction and add if within bounds
        # Checks that each x and y are within range for modification
        if x > 0:
            lx = x-1
            adjacent_indices.append(type + str(lx) + str(y))  # Left
        if x < max_x - 1:
            rx = x+1
            adjacent_indices.append(type + str(rx) + str(y))  # Right
        if y > 0:
            uy = y-1
            adjacent_indices.append(type + str(x) + str(uy))  # Up
        if y < max_y - 1:
            dy = y+1
            adjacent_indices.append(type + str(x) + str(dy))  # Down

        return adjacent_indices

    
    """ This function takes the position on the grid as arguments. It calculates the adjacent indicies 
    for that location. 
    Param:
        - x (int): grid row of current position
        - y (int): grid column of current position
    Return:
        - a list of adjacent indicies
    """
    def get_adjacent_indices(self, x, y):
        if not (0 <= x < 4 and 0 <= y < 4):
            raise ValueError("x and y must be within the grid boundaries (0 to 3).")
       
        max_x = 4 
        max_y = 4
        adjacent_indices = []
        # Check each direction and add if within bounds
        # Checks that each x and y are within range for modification
        if x > 0:
            lx = x-1
            adjacent_indices.append(str(lx) + str(y))  # Left
        if x < max_x - 1:
            rx = x+1
            adjacent_indices.append(str(rx) + str(y))  # Right
        if y > 0:
            uy = y-1
            adjacent_indices.append(str(x) + str(uy))  # Up
        if y < max_y - 1:
            dy = y+1
            adjacent_indices.append(str(x) + str(dy))  # Down

        return adjacent_indices
    
    
    # Since there is no percept for location, the agent has to predict
    # what location it is in based on the direction it was facing
    # when it moved

    def calculateNextPosition(self,action):
        if self.facing=='up':
            self.position = (self.position[0],max(0,self.position[1]-1))
        elif self.facing =='down':
            self.position = (self.position[0],min(self.max-1,self.position[1]+1))
        elif self.facing =='right':
            self.position = (min(self.max-1,self.position[0]+1),self.position[1])
        elif self.facing =='left':
            self.position = (max(0,self.position[0]-1),self.position[1])
        return self.position

    # and the same is true for the direction the agent is facing, it also
    # needs to be calculated based on whether the agent turned left/right
    # and what direction it was facing when it did
    
    def calculateNextDirection(self,action):
        if self.facing=='up':
            if action=='left':
                self.facing = 'left'
            else:
                self.facing = 'right'
        elif self.facing=='down':
            if action=='left':
                self.facing = 'right'
            else:
                self.facing = 'left'
        elif self.facing=='right':
            if action=='left':
                self.facing = 'up'
            else:
                self.facing = 'down'
        elif self.facing=='left':
            if action=='left':
                self.facing = 'down'
            else:
                self.facing = 'up'

    # this is the function that will pick the next action of
    # the agent. This is the main function that needs to be
    # modified when you design your new intelligent agent
    # right now it is just a random choice agent
    def action(self):

        # test for controlled exit at end of successful gui episode
        if self.stopTheAgent:
            print("Agent has won this episode.")
            self.gold = True
            return 'exit' # will cause the episide to end
            
        #reflect action -- get the gold!
        if 'glitter' in self.percepts:
            print("Agent will grab the gold!")
            self.stopTheAgent=True
            return 'grab'
        
        if self.give_up:
            print("Agent has given up :(")
            return 'exit'
        
        alpha = self.prev_alpha # Default
        
        # if rotate is true, skip finding alpha and direction
        if not self.rotate:
            # When adding Q_TABLE, need to have both model_enumeration and direction for planning part 
            # Find alpha from model checking
            mc_alpha = self.model_enumeration()
            
            # Find alpha based on MC or Q-table (prob of epsilon)
            alpha = self.choose_action(mc_alpha)
            
            # Set previous alpha for later -- if rotations are needed
            self.prev_alpha = alpha
            
            # choose direction based on action, and move          
            action = self.get_turn_direction(self.position, alpha, self.facing)
            
            if self.prev_state[0] is not None: # if action is None, we are still at start position and have not moved
                #if action is not None: # only None when first starting the game
                current_action = self.determine_action(self.position, alpha)
                probability = self.prob_safe[alpha]
                current_prob = self.get_bucket(probability) 
                #current_prob = self.get_bucket(self.prob_safe[alpha])
                current_state = (int(self.position[0]), int(self.position[1]), current_prob, current_action)
                self.sarsa_update(current_state)
            
            # Set state for next iteration
            prev_action = self.determine_action(self.position, alpha)
            
            # gets probs of all alphas in prob_safe, so should be 
            # fine if q-table chooses different alpha than prob MC
            probability = self.prob_safe[alpha]
            prev_prob = self.get_bucket(probability) 
            
            self.prev_state[0] = self.position[0]
            self.prev_state[1] = self.position[1]
            self.prev_state[2] = prev_prob
            self.prev_state[3] = prev_action
        else: # Rotation occured
            action = self.get_turn_direction(self.position, self.prev_alpha, self.facing)
        
        if action == 'move':
            # predict the effect of this
            self.calculateNextPosition(action)
        else:
            self.calculateNextDirection(action)
        print ("Intelligent agent:",action, "-->",self.position[1],
               self.position[0], self.facing)
        print('PREVIOUS STATE: ', self.prev_state)
        return action
            
    """
    This function find the next alpha based on probabilistic model checking
    Return:
        - alpha (string): next location with highest probability of being safe
    """
    def model_enumeration(self):
        adj_rooms = self.order_alpha() # Unexplored alphas first
        alpha = adj_rooms[-1] # default is the last room on the list
        found_next_action = False
        self.model_count = 0
        self.alpha_count = 0
        alpha_prob = {} # Probability of each of the alphas in this state
        safe_list = [] # List of 100% safe alphas
        for option in adj_rooms:
            if self.visited.count(option) < 4: # Only explore rooms 4 times - need to move on
                # Reset count for each alpha
                self.model_count = 0
                self.alpha_count = 0
                # Get symbols for model checking
                symbols = self.add_symbols(self.position)
                alpha = option
                if self.modelcheck(symbols, [], alpha):
                    found_next_action = True
                    self.prob_safe[alpha] = self.alpha_count / self.model_count if self.model_count > 0 else -1
                    alpha_prob[alpha] = self.alpha_count / self.model_count if self.model_count > 0 else -1
                    safe_list.append(alpha)
                else:
                    self.prob_safe[alpha] = self.alpha_count / self.model_count if self.model_count > 0 else -1
                    alpha_prob[alpha] = self.alpha_count / self.model_count if self.model_count > 0 else -1
        if len(safe_list) > 0: # 100% safe
            return safe_list[0]
        elif len(alpha_prob) > 0: # Find most safe room
            return max(alpha_prob, key=alpha_prob.get)
        else: # There are no more actions for the agent
            self.give_up = True
            return alpha

    """
    This function orders the alphas based on how many times it was visited.
    This helps to encourage the agent to move to unexplore rooms.
    Return:
        - ordered_rooms (list): list of adjacent rooms ordered from unexplored to 
        explored
    """
    def order_alpha(self):
        adj_rooms = self.get_adjacent_indices(self.position[0], self.position[1])
        ordered_rooms = []
        for room in adj_rooms:
            if room not in self.visited:
                ordered_rooms.append(room)  
        for room in adj_rooms:
            if room not in ordered_rooms:
                ordered_rooms.append(room)
        return ordered_rooms

    """
    Calculates the direction the agent needs to be in for next move. Validates that current and
    target positions are in range.
    Param:
        - current_pos: current location
        - target_pos: location of target position
        - current_facing: direction the agent is facing
    Return:
        - action (string): action agent need to take based on parameters
    """
    def get_turn_direction(self, current_pos, target_pos, current_facing):
        if not (0 <= int(target_pos[0]) < 4 and 0 <= int(target_pos[1]) < 4):
            raise ValueError("target x and y must be within the grid boundaries (0 to 3).")
        if not (0 <= int(current_pos[0]) < 4 and 0 <= int(current_pos[1]) < 4):
            raise ValueError("current x and y must be within the grid boundaries (0 to 3).")

        dx = int(target_pos[0]) - int(current_pos[0])
        dy = int(target_pos[1]) - int(current_pos[1])

        # Determine the required facing direction to move directly towards the target
        if dx > 0:
            required_facing = 'right'
        elif dx < 0:
            required_facing = 'left'
        elif dy > 0:
            required_facing = 'down'
        elif dy < 0:
            required_facing = 'up'
        else:
            required_facing = current_facing  # If no movement is needed

        # Map of directions to their left and right turns
        directions = {
            'up': {'left': 'left', 'right': 'right'},
            'down': {'left': 'right', 'right': 'left'},
            'left': {'left': 'down', 'right': 'up'},
            'right': {'left': 'up', 'right': 'down'}
        }

        # Decide turn based on current facing and required facing
        if current_facing == required_facing:
            self.rotate = False
            return 'move'  # No turn needed
        elif directions[current_facing]['left'] == required_facing:
            self.rotate = True
            return 'left'
        elif directions[current_facing]['right'] == required_facing:
            self.rotate = True
            return 'right'

        # If neither left nor right direct turns are required, a 180 turn is needed
        self.rotate = True
        return 'right'
 
    """
    Recursively enumerate all possible models and check if the models that satisfy KB are a subset of those
    that satisfy alpha. KB entails alpha implies M(KB) subset M(alpha)
    Param:
        - symbols (list): All the symbols for the model
        - model (list): list of symbols with their cooresponding truth value
        - alpha (string): the next possible location
    """
    def modelcheck(self, symbols, model, alpha):
        if len(symbols) == 0:
            # When all symbols have been assigned, check if the model is consistent
            if self.isTrue(model):   
                self.model_count += 1
                # Only if the model is valid, check if alpha holds true in this model
                if self.is_alpha_true(alpha, model):
                    self.alpha_count += 1
                return self.is_alpha_true(alpha, model)
            else:
                return True  # The model is not valid, so return False without checking alpha
        else:
            p = symbols[0]
            rest = symbols[1:]  # Remaining symbols

            # Explore both possibilities for the current symbol (true and false)
            # and ensure both sides of the branch are consistent before accepting them
            result_if_true = self.modelcheck(rest, model + [(p, True)], alpha) 
            result_if_false = self.modelcheck(rest, model + [(p, False)], alpha) 

            # Return true if any of the recursive calls find a valid model where alpha is true
            return result_if_true and result_if_false

    """
    Checks if a model adheres to the rules of the game and is consistant to the current knowledge
    Param:
        - model (list): list of symbol and truth value pairs
    Return:
        - True if model is both consistant and adhere to the rules of the game, 
        False otherwise
    """
    def isTrue(self, model):
        return self.correct_adj_rooms(model) and self.model_consistency(model)
            

    """
    Checks if model adheres to the rules of the game.
    Param:
        - model (list): list of symbol and truth value pairs
    Return:
        - True if model is adheres to the rules of the game, 
        False otherwise
    """
    def correct_adj_rooms(self, model):
        wumpus_count = 0
        model_dict = dict(model)  # Convert list of tuples to dictionary for easy lookup

        for room, truth_val in model:
            room_type = room[0]
            room_row = int(room[1])
            room_col = int(room[2])
            adj_rooms = self.get_adjacent_indices(room_row, room_col)

            if truth_val:
                if room_type == 'b':  # Breeze implies at least one adjacent pit
                    # if there is not any true pit surround a breeze, and there are any pit rooms in the model
                    pit_found = False
                    pit_exist = False
                    for r in adj_rooms:
                        if ('p' + str(r[0]) + str(r[1])) in model_dict: 
                            pit_exist = True
                            if ('p' + str(r[0]) + str(r[1]), True) in model:
                                pit_found = True 
                    # Checks if there are pit symbols in the model and if so, do they adhere to the rules?
                    if not pit_found and pit_exist: 
                        return False
                elif room_type == 's':  # Stench implies at least one adjacent Wumpus
                    wumpus_found = False
                    wumpus_exist = False
                    wumpus_count = 0
                    for r in adj_rooms:
                        if ('w' + str(r[0]) + str(r[1])) in model_dict:
                            wumpus_exist = True
                            if ('w' + str(r[0]) + str(r[1]), True) in model:
                                wumpus_found = True 
                                wumpus_count += 1
                    # Checks if there are wumpus symbols in the model and if so, do they adhere to the rules?
                    if not wumpus_found and wumpus_exist:
                        return False
                    elif wumpus_count > 1: # Can only have one wumpus
                        return False
                elif room_type == 'p':  # Pit should cause a breeze in all adjacent rooms known to be in the model
                    for r in adj_rooms:
                        if ('b' + str(r[0]) + str(r[1]), False) in model:
                            return False
                elif room_type == 'w':  # Wumpus should cause a stench in all adjacent rooms known to be in the model
                    for r in adj_rooms:
                        if ('s' + str(r[0]) + str(r[1]), False) in model:
                            return False

        # Ensure there is only one Wumpus in the model
        if wumpus_count > 1:
            return False

        return True


    """
    Checks that the model is consistant with the map
    Param:
        - model (list): list of symbol and truth value pairs
    Return:
        - True if model is consistant, 
        False otherwise
    """
    def model_consistency(self, model):
        model_dict = dict(model)
        # We know the room is safe and do not need to check models that mark the room as not safe
        for room in self.safe:
            if ('p'+room, True) in model or ('w'+room, True) in model:
                return False
        # Iterate through each cell in the map
        for i, row in enumerate(self.map):
            for j, cell in enumerate(row):
                # Check stench consistency
                stench_key = 's' + str(i) + str(j)
                if cell[0] is None and (stench_key, True) in model:
                    return False
                if cell[0] and stench_key in model_dict:
                    if (stench_key, False) in model:
                        return False
                
                # Check breeze consistency
                breeze_key = 'b' + str(i) + str(j)
                if cell[1] is None: 
                    if (breeze_key, True) in model:
                        return False
                if cell[1] and breeze_key in model_dict:
                    #print(breeze_key)
                    if (breeze_key, False) in model:
                        return False
        return True

    """
    Checks if alpha is true in the model - no pit or wumpus in the desired location
     - model (list): list of symbol and truth value pairs
    Return:
        - True if model is adheres to the rules of the game, 
        False otherwise
    """
    def is_alpha_true(self, alpha, model):
        pit = 'p' + alpha
        wumpus = 'w' + alpha
        if (pit, True) in model or (wumpus, True) in model:
            return False
        return True

    def add_symbols(self, position):
        if not (0 <= int(position[0]) < 4 and 0 <= int(position[1]) < 4):
            raise ValueError("position x and y must be within the grid boundaries (0 to 3).")
        adj_rooms = self.get_adjacent_indices(position[0], position[1])
        symbols = []
        symbols.append('p'+str(position[0]) + str(position[1]))
        symbols.append('w'+str(position[0]) + str(position[1]))
        symbols.append('s'+str(position[0]) + str(position[1]))
        symbols.append('b'+str(position[0]) + str(position[1]))
        for room in adj_rooms:
            symbols.append('p'+room)
            symbols.append('w'+room)
            symbols.append('s'+room)
            symbols.append('b'+room)
        for room_id in self.safe:
            if 'p'+str(room_id) in symbols:
                symbols.remove('p'+str(room_id))
            if 'w'+str(room_id) in symbols:
                symbols.remove('w'+str(room_id))
        return symbols

    """ 
    Perform a SARSA update for a single state transition. 
    Param:
        - next_state (tuple): contains the x, y, probability bucket and action for
        the next state
    """
    def sarsa_update(self, next_state):
        print('UPDATING SARSA')
        reward = -1 # Default reward -- when episode ends, sarsa_update_terminal is called 
                
        # Current Q-value
        # X, Y, Prob bucket, and action
        current_q = self.q_table[int(self.prev_state[0]), int(self.prev_state[1]), int(self.prev_state[2]), int(self.prev_state[3])]
        # Q-value for the next state-action pair
        # X, Y, Prob bucket, and action
        next_q = self.q_table[int(next_state[0]), int(next_state[1]), int(next_state[2]), int(next_state[3])]

        # SARSA update formula
        learning = self.learning_rate * (reward + self.discount * next_q - current_q)
        updated_val = current_q + learning
        self.q_table[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3]] += self.learning_rate * (reward + self.discount * next_q - current_q)
  
    """
    This updates the Q-Table when at a terminal state
    """  
    def sarsa_update_terminal(self):
        reward = -1 # Default reward - if the agent gave up
        if self.dead: # If the agent died, large negative reward
            reward = -1000
        elif self.gold: # If the agent won, large positive reward
            reward = 1000

        # Current Q-value
        # X, Y, Prob bucket, and action
        current_q = self.q_table[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3]]
        
        # SARSA update formula
        self.q_table[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3]] += self.learning_rate * (reward - current_q) 
        
    """ 
    This calculates the bucket index for a given probability
    Param:
    - probability (float): probability of a given position
    Return:
        - bucket index (0, 1, 2, or 3)
    """
    def get_bucket(self, probability):
        # Determine the appropriate bucket for the given probability
        if probability == 1.0:
            return 3
        else:
            return np.digitize(probability, self.buckets) - 1

    """
    Determine the action needed to move from current_position to next_position.
    Parameters:
        - current_position (tuple): The current coordinates as (x, y).
        - next_position (tuple): The next coordinates as (x, y).
    
    Returns:
        -  str: The action 'up', 'down', 'left', or 'right' needed to move to the next_position,
        or 'none' if no single-step action is appropriate.
    """
    def determine_action(self, current_position, next_position):
        if not (0 <= int(next_position[0]) < 4 and 0 <= int(next_position[1]) < 4):
            raise ValueError("next x and y must be within the grid boundaries (0 to 3).")
        if not (0 <= int(current_position[0]) < 4 and 0 <= int(current_position[1]) < 4):
            raise ValueError("current x and y must be within the grid boundaries (0 to 3).")
        # Calculate differences between the positions
        dx = int(next_position[0]) - int(current_position[0])
        dy = int(next_position[1]) - int(current_position[1])

        # Determine the action based on coordinate differences
        if dx == 1 and dy == 0:
            return 3
        elif dx == -1 and dy == 0:
            return 2
        elif dx == 0 and dy == 1:
            return 1
        elif dx == 0 and dy == -1:
            return 0

    """
    choose_action uses epsilon-greedy policy to decide which action is 
    taken from the current position. Choosing model checking has a higher
    probability throughout all episodes. 
    Param:
        - alpha (string): the next location found from probabilistic model checking
    Return:
        - alpha (string): the choosen action given the epsilon-greedy policy
    """
    def choose_action(self, alpha):
        if np.random.rand() > self.epsilon: # Probability of choosing planning
            # Count the number of times planning is picked
            self.mc_count += 1
            
            # Return the next location 
            # Since model checking has to run for each move to generate probabilities, we
            # avoid unnecessarily running it again
            return alpha
            
        else: # Choose action with best value from q-table
            # Get Q-values for the given state and probability
            self.give_up = False # If RL is choosen, then the agent wants to keep exploring
            # Count the number of times learning is picked
            self.rl_count += 1
            
            # Get the probability and bucket for the current position
            prob = self.prob_safe[str(self.position[0]) + str(self.position[1])]
            bucket = self.get_bucket(prob)
            
            # The q_values for the current position and probability bucket and all actions
            q_values = np.copy(self.q_table[self.position[0], self.position[1], bucket, :])
            
            # Select the action with the highest Q-value
            max_action_index = self.max_action(self.position[0], self.position[1], q_values)
            
            # Get the mapping from direction index to actual direction
            q_action = self.directions.get(max_action_index)
            
            # Find the alpha location based on the action 
            alpha = self.move_position(self.position, q_action)
            #print(f'SARSA. Chosen Alpha is {alpha} with direction {q_action}')
            return alpha
            
            
    """
    max_action finds the max move for a given set of q_values. 
    It insures that the returned action is within the bounds of
    the grid by setting invalid moves to negative infinity. 
    Param: 
        - x (int): current row position
        - y (int): current column position
        - q_values (list) - contains the q-values of the x,y, probability
            bucket of the current position for each action. 
     Return: action_index - index of the max valid action
     """
    def max_action(self, x, y, q_values):
        if not (0 <= x < 4 and 0 <= y < 4):
            raise ValueError("x and y must be within the grid boundaries (0 to 3).")
       
        # Define movements associated with each action
        movements = {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0)    # right
        }

        # Disable actions that would move out of bounds
        valid_actions = False  # Flag to check if there is at least one valid action
        for action in range(4):
            new_x, new_y = x + movements[action][0], y + movements[action][1]
            if new_x < 0 or new_x >= self.max or new_y < 0 or new_y >= self.max:
                q_values[action] = -np.inf  # Set to negative infinity to ignore this action
            else:
                valid_actions = True  # Found at least one valid action

        # Handle case where no actions are valid
        if not valid_actions:
            print("Warning: No valid actions available.")
            return None  # or handle this case as needed in your simulation

        # Select the action index with the highest Q-value
        action_index = np.argmax(q_values)
        return action_index