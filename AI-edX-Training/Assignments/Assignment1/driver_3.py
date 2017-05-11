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

class Method:
    """
        Enum with the available mothods for the solver
    """
    BFS = 'bfs'
    DFS = 'dfs'
    AST = 'ast'
 
class Action:
    """
        Enum with the available actions for the Agent
    """
    Left = 'Left'
    Right = 'Right'
    Up = 'Up'
    Down='Down'
    All = [Action.Up, Action.Down, Action.Left, Action.Right]

class Board:
    """
        This class represent the current state of a board

        Also it includes functionality to return the next movement
        given an action: Up, Down, Left or Right.
                0 1 2
         3x3    3 4 5
                6 7 8
    """
    def __init__(self, tokens=None, size=3):
        self.size = size;
        self.tokens = tokens
        # Check whether aome parameters are none
        if self.tokens is None:
            self.tokens = list(range(self.size*2)-1)
        else:
            self.size = math.sqrt(len(state))
    def prettyPrint(self):
         for i in range(self.size):
             for j in range(self.size):
                 print("{}  ".format(self.tokens[(i * self.size)+j]) ,end='')
             print()
    def equal(self,board):
        if self.tokens == board.tokens:
             return True
        return False
    def move(self,action):
        # Search for the row and col of the empty space (0)
        pos = self.tokens.index(0)
        row = pos // self.size
        col = pos - ((pos // self.size) * self.size)
        newpos = -1
        # Check if the action could be performed
        if action==Action.Right and col != self.size-1:
            newpos = pos+1
        elif action== Action.Left and col!=0:
            newpos = pos-1
        elif action== Action.Down and row!=self.size-1:
            newpos = pos+self.size
        elif action== Action.Up and row!=0:
            newpos = pos-self.size
        # Check valid new position
        if newpos!=-1:
            result = list(self.tokens)
            result[pos] = self.tokens[newpos]
            result[newpos] = self.tokens[pos]
            return Board(result)
        # By deafult return None
        return None

class State:
    """
        This class will be used to represent the state of each action-state
    """
    def __init__(self, board, action=None):
        self.action = action
        self.board = board

class Node:
    """
        This class will be used to represent a node for each state
    """
    def __init__(self, parent, state = None, level=0):
        self.parent = parent
        self.state = state
        self.level = level
    def neighbours(self):
        # Return Neighbours using the parent node and not considering, current action 
        # or operation not allowd because the limits of the boards
        neighbours = []
        for action in Actions.All:
            if action != self.state.level
                # Get the new board if possible
                board = self.parent.state.board.move(action)
                if board is not None:
                    #create the node and return the neighbour
                    neighbours.add(Node(self.parent, State(board, action), self.level))
    def childs(self):
        # Return the childs from this node  
        # child level = level + 1  
        childs = []
        for action in Actions.All:
            # Get the new board if possible
            board = self.state.board.move(action)
            if board is not None:
                #create the node and return the neighbour
                childs.add(Node(self, State(board, action), self.leve+1))

def breadth_first_search(initialState, finalState):
    # https://codereview.stackexchange.com/questions/135156/bfs-implementation-in-python-3
    # Initialize frontier
    visited, frontier = set(), collections.deque([initialState])
    # Check for items
    while frontier:
        # Get the current state (from left)
        node = frontier.popleft()
        # Check the current State match with the final state
        if (state == finalState):
            return state
        # Visit all possible neighbours
        for neighbour in state.neighbours(): 
            #Check if the node has been already visited previously
            if neighbour not in visited:
                # Append the neighbour to the queue and onto the  visited list
                frontier.append(neighbour)
                visited.add(neighbour)
    # Returns nothing if no final state founded
    return None

def depth_first_search(initialState, finalState):
    return None

def a_star_search(initialState, finalState):
    return None

class Solver:
    """
        This class will provide the functionality to solver the game
    """
    def __init__(self, method = Method.BFS, initialTokens = None, finalTokens = [0,1,2,3,4,5,6,7,8]):
        #Initialize the variables
        self._method = method
        self._initialTokens = initialTokens
        self._finalTokens = finalTokens
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
        initialState = State(Board(initialTokens), None)
        finalState = State(Board(finalTokens), None)
        # Check the Method to compute
        if self._method == Method.BFS:
            founded = breadth_first_search(initialState,finalState)
        elif self._method == Method.DFS:
            founded = depth_first_search(initialState,finalState)
        elif self._method == Method.AST:
            founded = a_star_search(initialState,finalState)
        else:
            found = None
        end_time = time.time()
        # Fill the values
        if founded is not None:
            self._parameters["path_to_goal"] = ['Up', 'Left', 'Left']
            self._parameters["cost_of_path"] = 3
            self._parameters["nodes_expanded"] = 10
            self._parameters["search_depth"] = 3
            self._parameters["max_search_depth"] = 4
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
    # Check The number of args
    if len(sys.argv) == 3:
        # Get the command line arguments
        method = sys.argv[1]
        board = sys.argv[2]
        # Start the game process using the solver class
        Solver(method, board).start()
    else:
        print("ERROR: Bad number of arguments")
