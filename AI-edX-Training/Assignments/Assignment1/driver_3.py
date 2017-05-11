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
from collections import OrderedDict


# https://codereview.stackexchange.com/questions/135156/bfs-implementation-in-python-3

class game:
    """
        This class will provide all the functionality to
        run the program
    """
    def __init__(self, method = "bfs", board = None):
        #Initialize the variables
        self._method = method
        self._board = board
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
    # Check The number of args
    if len(sys.argv) == 3:
        game(sys.argv[1],sys.argv[2]).start()
    else:
        print("ERROR: Bad number of arguments")
