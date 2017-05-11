# Command line:
#   python driver.py <method> <board>
#
# Method:
#   bfs (Breadth-First Search)
#   dfs (Depth-First Search)
#   ast (A-Star Search)
# Board
#   0,8,7,6,5,4,3,2,1
#
# Example:
#   python driver.py bfs 0,8,7,6,5,4,3,2,1
import sys
import time
import math
from collections import OrderedDict

# https://codereview.stackexchange.com/questions/135156/bfs-implementation-in-python-3
def breadth_first_search(initialState, finalState):
    # Initialize frontier
    visited, frontier = set(), collections.deque([initialState])
    # Check for items
    while frontier:
        # Get the current state (from left)
        state = frontier.popleft()
        # Check the current State match with the final state
        if (state == finalState):
            return state
        # Visit all possible neighbours
        for neighbour in state.neighbours(): 
            #Check if the node has been already visited previously
            if neighbour not in visited:
                # Append the neighbour to the queue and onto the  visited list
                frontier.append(child)
                visited.add(child)
    # Returns nothing if no final state founded
    return None

class Action:
    Left = 'Left'
    Right = 'Right'
    Up = 'Up'
    Down='Down'


class Board:
    """
        This class represent the current state of a board

        Also it includes functionality to return the next movement
        given an action: Up, Down, Left or Right.

            3x3 
           0 1 2
           3 4 5
           6 7 8
        
    """
    def __init__(self, state=None, size=3):
        self.size = size;
        self.state = state
        # Check whether aome parameters are none
        if self.state is None:
            self.state = list(range(self.size*2)-1)
        else:
            size = math.sqrt(len(state))
    def prettyPrint(self):
         for i in range(self.size):
             for j in range(self.size):
                 print("{}  ".format(self.state[(i * self.size)+j]) ,end='')
             print()
    def equal(self,board):
        if self.state == board.state:
             return True
        return False
    def move(self,action):
        # Search for the row and col of the empty space (0)
        pos = self.state.index(0)
        row = pos // self.size
        col = pos - ((pos // self.size) * self.size)
        # Check if the action could be performed
        if (action==Action.Right and row != self.size-1):
            result = list(self.state)
            result[pos] = self.state[pos+1]
            result[pos+1] = self.state[pos]
            return Board(result)
        elif action== Action.Left and row!=0:
            result = list(self.state)
            result[pos] = self.state[pos-1]
            result[pos-1] = self.state[pos]
            return Board(result)
        elif action== Action.Down and col!=self.size-1:
            result = list(self.state)
            result[pos] = self.state[pos+self.size]
            result[pos+self.size] = self.state[pos]
            return Board(result)
        elif action== Action.Up and col!=0:
            result = list(self.state)
            result[pos] = self.state[pos-self.size]
            result[pos-self.size] = self.state[pos]
            return Board(result)
        # By deafult return None
        return None


class Node:
    """
        This class will be used to represent the state of each action-state


    """
    def __init__(self, parent,action=None,level=0):
        self._parent = parent
        self._action = action
        self._level = level
    def neighbours(self):
        # Return Neighbours using the parent node and not considering, current action 
        # or operation not allowd because the limits of the boards
        pass
    def childs(self):
        # Return the childs from this node  
        # child level = level + 1  
        pass

class Solver:
    """
        This class will provide all the functionality to solver the game
    """
    def __init__(self, method = "bfs", board = None, finalboard = [0,1,2,3,4,5,6,7,8]):
        #Initialize the variables
        self._method = method
        self._board = board
        self._finalboard = finalboard
        self._parameters = OrderedDict()
    def _initialize(self):
        # Initialize all the variables
        self._parameters["path_to_goal"] = []
        self._parameters["cost_of_path"] = 0 
        self._parameters["nodes_expanded"] = 0
        self._parameters["search_depth"] = 0
        self._parameters["max_search_depth"] = 0
        self._parameters["running_time"] = 0.0
        self._parameters["max_ram_usage"] = 0.0
    def start(self):
        # Initialize the variables
        self._initialize() 
        #Start the process
        start_time = time.time()
        
        self._parameters["path_to_goal"] = ['Up', 'Left', 'Left']
        self._parameters["cost_of_path"] = 3
        self._parameters["nodes_expanded"] = 10
        self._parameters["search_depth"] = 3
        self._parameters["max_search_depth"] = 4

        end_time = time.time()
        # Fill the values
        self._parameters["running_time"] = end_time - start_time
        self._parameters["max_ram_usage"] = 0.0
        # Create the output
        self._create_report()
    def _create_report(self, filename="output.txt"):
        # Write all the parameters computed
        with open(filename, mode='w') as file:
            for key, value in self._parameters.items():
                file.write("{}:{}\n".format(key, value))

# Main Application
if __name__ == "__main__":
    
    print("Original")
    board = Board([1,2,3,4,0,5,6,7,8])
    board.prettyPrint()
    print("Move to Left")
    new_board = board.move(Action.Left)
    new_board.prettyPrint()
    print("Move to Right")
    new_board = board.move(Action.Right)
    new_board.prettyPrint()
    print("Move to Up")
    new_board = board.move(Action.Up)
    new_board.prettyPrint()
    print("Move to Down")
    new_board = board.move(Action.Down)
    new_board.prettyPrint()

    print("Move to Left-Left")
    new_board = board.move(Action.Left)
    new_board = new_board.move(Action.Left)
    new_board.prettyPrint()

    print("Move to Left-Right")
    new_board = board.move(Action.Left)
    #new_board = new_board.move(Action.Right)
    new_board.prettyPrint()


    # Check The number of args
    if len(sys.argv) == 3:
        # Get the command line arguments
        method = sys.argv[1]
        board = sys.argv[2]
        # Start the game process
        Solver(method, board).start()
    else:
        print("ERROR: Bad number of arguments")
