import sys
import math
from random import randint
from BaseAI_3 import BaseAI

adjacent_tiles = ((-1,0),(0,1),(1,0),(0,-1))

def get_smoothness(grid):
    """ Get the smoothness (score) for the current grid state

    The idea is to lopp over the grid and see the tiles adjacent
    similar tu the current one and not empty.

    """
    score = 0
    for x in range(grid.size):
        for y in range(grid.size):
            # Get the current value x,y
            current_value = grid.getCellValue((x,y))
            # Check not is empty
            if current_value != 0:
                # Check the adjacent cells and crossBound
                for tile in adjacent_tiles:
                    tile_pos = (x+tile[0],y+tile[1])
                    # Check if the current pos is in the bounds
                    if not grid.crossBound(tile_pos) and grid.getCellValue(tile_pos) == current_value:
                        #Comparet with the valie
                        score += 1
    # Finally return the total score founded
    return score

def rotate_grid(grid):
    matrix = []
    for y in range(grid.size):
        column=[]
        for x in range(grid.size):
            column.append(grid.map[x][y])
        matrix.append(column)
    return matrix      

def get_monotonicity(grid):
    """ Get the monotonicity (score) for the current grid state
    """
    # This will return the sum computed
    monotonicity_axis = []
    # Check from (0,0) too (3,3)
    for x in range(grid.size):
        array = [True if first>=second else False for first, second in zip(grid.map[x],grid.map[x][1:])]
        monotonicity_axis.append(all(array))
    # Rotate the grid and do the same
    rotate = rotate_grid(grid)
    for x in range(grid.size):
        array = [True if first>=second else False for first, second in zip(rotate[x],rotate[x][1:])]
        monotonicity_axis.append(all(array))
    # Return the sum for all the monotonocity being true
    return all(monotonicity_axis)

def get_heuristic(grid):
    """ Get the heuristic (score) for the current grid state
    """
    # Actual score is the max value merged in the grid
    actual_score = grid.getMaxTile()
    # Free Tiles available (more the better)
    available_cells = len(grid.getAvailableCells())
    # Values decreasing or decreasing along the edges.
    monotonicity_score = get_monotonicity(grid)
    # Smoothnes measure same values for adjacent tiles
    smoothness_score = get_smoothness(grid)
    # clustering_score 
    clustering_score = (monotonicity_score * 1000) + smoothness_score

    # Create an heuristic with the previous parameters
    return int(actual_score + math.log(actual_score) * available_cells + clustering_score)

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

       

