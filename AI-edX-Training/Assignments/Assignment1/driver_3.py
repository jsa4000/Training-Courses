#######################################
#  Week 2 Project: Search Algorithms  #
#######################################
# Command line:
#   python driver.py <method> <board>
#   Where:
#       Method. Algorithm that will be used
#           "bfs" (Breadth-First Search)
#           "dfs" (Depth-First Search)
#           "ast" (A-Star Search)
#       Board. Initial board state. ej "0,8,7,6,5,4,3,2,1"
#
#   Example: python driver.py bfs 0,8,7,6,5,4,3,2,1
import sys
import time
import math
from collections import OrderedDict, deque

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
    All = [Up, Down, Left, Right]

def toHash(x):
    return "{}".format(x)

class Board:
    """
        This class represent the current state of a board

        Also it includes functionality to return the next movement
        given an action: Up, Down, Left or Right.
        ------------------------------------------------------------
                | 0 1 2 |   move(Right)   | 1 0 2 |  
         3x3    | 3 4 5 |       =>        | 3 4 5 | 
                | 6 7 8 |                 | 6 7 8 |
        ------------------------------------------------------------        
    """
    def __init__(self, tokens=None, size=3):
        self.size = size;
        self.tokens = tokens
        # Check whether aome parameters are none
        if self.tokens is None:
            self.tokens = list(range(self.size*2)-1)
        else:
            self.size = int(math.sqrt(len(self.tokens)))
    def pretty_print(self):
         for i in range(self.size):
             print("| ",end='')
             for j in range(self.size):
                 print("{} | ".format(self.tokens[(i * self.size)+j]) ,end='')
             print()
    def equal(self,board):
        if self.tokens == board.tokens:
             return True
        return False
    def get_positions(self,action):
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
        return [pos, newpos]    
    def allowed(self,action):
        return self.get_positions(action)[1]>-1
    def move(self,action):
        pos, newpos = self.get_positions(action)
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
    def equal(self, state):
        if self.board.equal(state.board):
            return True
        return False

class Node:
    """
        This class will be used to represent a node for each state.
        It will be contain its parent node and the depth (level).
    """
    def __init__(self, parent=None, state=None, level=0):
        self.parent = parent
        self.state = state
        self.level = level
    def childs(self):
        # Return the childs from this node: child level = level + 1  
        childs = []
        for action in Action.All:
            # Get the new board if possible
            board = self.state.board.move(action)
            if board is not None:
                #create the node and return the neighbour
                childs.append(Node(self, State(board, action), self.level+1))
        # List comprenhension seems to work even worst.
        #return [Node(self, State(self.state.board.move(action), action), self.level+1) for action in Action.All if self.state.board.allowed(action)]
        return childs    
    def pretty_print(self):
        print("----------------------")
        print("  Level: {}".format(self.level))
        print("  Action: {}".format(self.state.action))
        print("  Board:")
        self.state.board.pretty_print() 
        print("----------------------")  

def breadth_first_search(initialState, finalState):
    # Initialize frontier
    visited, nodes = set(), deque([Node(None, initialState)])
    expanded, max_level = 0, 0
    # Check for items
    while nodes:
        # Get the first element added (from left)
        node = nodes.popleft()
        # Check the current State match with the final state
        if (node.state.equal(finalState)):
            return [node, expanded, max_level]
        # Expand the current node
        expanded += 1
        # Visit all possible neighbours
        for child in node.childs(): 
            #Check if the node has been already visited previously
            if toHash(child.state.board.tokens) not in visited:
                if (child.level>max_level):
                    max_level = child.level
                # Append the neighbour to the queue and onto the  visited list
                nodes.append(child)
                visited.add(toHash(child.state.board.tokens))
    # Returns nothing if no final state founded
    return None

def depth_first_search(initialState, finalState):
    # Initialize frontier
    visited, nodes = set(), deque([Node(None, initialState)])
    expanded, max_level = 0, 0
    # Check for items
    while nodes:
        # Get the first element added (from left)
        node = nodes.pop()
        # Check the current State match with the final state
        if (node.state.equal(finalState)):
            return [node, expanded, max_level]
        # Expand the current node
        expanded += 1
        # Visit all possible neighbours
        for child in node.childs()[::-1]: 
            #Check if the node has been already visited previously
            if toHash(child.state.board.tokens) not in visited:
                if (child.level>max_level):
                    max_level = child.level
                # Append the neighbour to the queue and onto the  visited list
                nodes.append(child)
                visited.add(toHash(child.state.board.tokens))
    # Returns nothing if no final state founded
    return None

def a_star_search(initialState, finalState):
    # Initialize frontier
    visited, nodes = set(), list([Node(None, initialState)])
    expanded, max_level = 0, 0
    # Check for items
    while nodes:
        # Sort the queue by thes heuristic computed
        nodes.sort(key = heuristic)
        # Get the first element sorted (from left)
        node = nodes.pop(0)
        # Check the current State match with the final state
        if (node.state.equal(finalState)):
            return [node, expanded, max_level]
        # Expand the current node
        expanded += 1
        # Visit all possible neighbours
        for child in node.childs(): 
            #Check if the node has been already visited previously
            if toHash(child.state.board.tokens) not in visited:
                if (child.level>max_level):
                    max_level = child.level
                # Append the neighbour to the queue and onto the  visited list
                nodes.append(child)
                visited.add(toHash(child.state.board.tokens))
    # Returns nothing if no final state founded
    return None

def heuristic(x):    
    goal = list(range(len(x.state.board.tokens)))
    masks = [0 if j - i==0 else 1 for i,j in zip(x.state.board.tokens,goal)]
    return x.level + sum(masks) 

class Solver:
    """
        This class will provide the functionality to solver the game
    """
    def __init__(self, initialTokens, method = Method.BFS, finalTokens = None):
        #Initialize the variables
        self._method = method
        self._initialTokens = initialTokens
        self._finalTokens = finalTokens
        if self._finalTokens is None:
            self._finalTokens = list(range(len(initialTokens)))
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
    def _get_actions(self, node):
        path = []
        while node.state.action:
            path.insert(0,node.state.action)
            node = node.parent
        return path    
    def _create_report(self, filename="output.txt"):
        # Write all the parameters computed
        with open(filename, mode='w') as file:
            for key, value in self._parameters.items():
                file.write("{}:{}\n".format(key, value))
    def start(self):
        # Initialize the variables
        self._initialize() 
        #Start the process
        start_time = time.time()
        initialState = State(Board(self._initialTokens), None)
        finalState = State(Board(self._finalTokens), None)
        # Check the Method to compute
        if self._method == Method.BFS:
            node, expanded, max_level = breadth_first_search(initialState,finalState)
        elif self._method == Method.DFS:
            node, expanded, max_level = depth_first_search(initialState,finalState)
        elif self._method == Method.AST:
            node, expanded, max_level = a_star_search(initialState,finalState)
        else:
            node = None
        end_time = time.time()
        # Fill the values
        if node is not None:
            node.pretty_print()
            self._parameters["path_to_goal"] = self._get_actions(node)
            self._parameters["cost_of_path"] = node.level
            self._parameters["nodes_expanded"] = expanded - 1
            self._parameters["search_depth"] = node.level
            self._parameters["max_search_depth"] = max_level
        self._parameters["running_time"] = end_time - start_time
        self._parameters["max_ram_usage"] = 0.0
        # Create the output report
        self._create_report()
  
# Main Application
if __name__ == "__main__":
    # Check The number of args
    if len(sys.argv) == 3:
        # Get the command line arguments
        method = sys.argv[1]
        board = list(map(int,sys.argv[2].split(sep=",")))
        # Start the game process using the solver class
        Solver(board, method).start()
    else:
        print("ERROR: Bad number of arguments")
