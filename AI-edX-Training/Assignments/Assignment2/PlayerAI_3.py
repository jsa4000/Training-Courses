import sys
import math
from random import randint
from BaseAI_3 import BaseAI

def get_heuristic(grid):
    """ Get the heuristic (score) for the current grid state
    """
    actual_score = grid.getMaxTile()
    available_cells = len(grid.getAvailableCells())
    clustering_score = 1.0


    retur

def insert_random_tile(grid):
    """ Insert a value [2, 4] with a probability and added  
    into a random cell available in the grid
    """
    # Get a random vale with probability = 0.9 for 2
    if randint(0,99) < 100 * 0.9:
        tileValue = 2
    else:
        tileValue = 4;
    # Get current availabile cells remain in the grid
    cells = grid.getAvailableCells()
    # Select a random cell
    cell = cells[randint(0, len(cells) - 1)]
    # Set the cell selected with the value
    grid.setCellValue(cell, tileValue)

def alpha_beta(grid, depth, alpha, beta, maximize):
    """
    """
    # Get the available nodes. If any
    moves = grid.getAvailableMoves()
    # Check if final state or depth
    if len(moves) == 0 or depth == 0:
        return (get_heuristic(grid), None)
    # Now check the case to maximize or minimize
    if maximize:
        # Initialize the variables
        max_move = None
        max_value = -sys.maxsize
        # Go through all the moves allowed
        for move in moves:
            # Get a clone for the current move
            next_grid = grid.clone()
            next_grid.move(move)
            # Now lets add a random tile for computer player
            insert_random_tile(next_grid)
            # Get the move and score of the minimizer
            value, _ = alpha_beta(next_grid, depth-1, alpha, beta,False)
            # Now update the max value and the alpha if changes
            if value > max_value:
                max_value, max_move =  value, move
            # Check if we must to break the loop before update alpha
            if max_value >= beta: 
                break
            # Finally update alpha
            alpha = max(max_value, alpha)
        #return the current values for the maximizer
        return (max_value, max_move)
    else:
        # Initialize the variables
        min_move = None
        min_value = sys.maxsize
        # Go through all the moves allowed
        for move in moves:
            # Get a clone for the current move
            next_grid = grid.clone()
            next_grid.move(move)
            # Now lets add a random tile for computer player
            insert_random_tile(next_grid)
            # Get the move and score of the maximizer
            value, _ = alpha_beta(next_grid, depth-1, alpha, beta, True)
            # Now update the max value and the alpha if changes
            if value < min_value:
                min_value, min_move =  value, move
            # Check if we must to break the loop before update beta
            if min_value <= alpha: 
                break
            # Finally update beta
            beta = min(min_value, beta)
        #return the current values for the minimizer
        return (min_value, min_move)

class PlayerAI(BaseAI):
    
    # const tha will be used to optimize the algorithm 
    MAX_DEPTH = 5

    # https://sandipanweb.wordpress.com/2017/03/06/using-minimax-with-alpha-beta-pruning-and-heuristic-evaluation-to-solve-2048-game-with-computer/

    def getMove(self, grid):
        """ This function get the current move that maximize the
        score. This will use the minmax algortihm with alph-beta
        pruning.
        """
        # First we need a clone of the grid to perform the search
        grid_cloned = grid.clone()
        # Get the move that perform better with the current grid settings.
        _, move = alpha_beta(grid_cloned, PlayerAI.MAX_DEPTH, -sys.maxsize, sys.maxsize, True)
        #Return the current move performed or None if end.
        return move

       

