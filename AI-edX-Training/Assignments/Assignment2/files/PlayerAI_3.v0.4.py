import sys
import math
from random import randint
from BaseAI_3 import BaseAI

def get_heuristic_score(grid):
    """ This function gets the current score for the current
    grid. The score is independt of the minmax node, since the
    algorithm itself need to choose between the worst or better
    score given by this heuristic.

    There are several techniques to play better ar 2048 game. So,
    I will recopile all this techniques to maximize the score.

    The score must the scaled or fitted into a some interval, so
    the computation could be better in performances. In order
    to do this don't try to compute floating values or very huge
    values...

    """
    actual_score = grid.getMaxTile()
    available_cells = len(grid.getAvailableCells())
    clustering_score = 1.0

    # Compute the current scode
    score = int(actual_score+math.log(actual_score)*available_cells - clustering_score)
    return max(score, min(actual_score, 1))

def insert_random_tile(grid):
    """ Insert a random tile
    """
    # Get a random vale with probability = 0.9
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

def alpha_beta(grid, depth, alpha, beta, maximize=True):
    """ This is the max search algoritm for the minmax. 
    Prior this movements the addictional tiles 2 or 4 must be
    inserted after the next movement or iteration
    """
    # First get the current available moves to check if leaf or not
    moves = grid.getAvailableMoves()
    # Check if we are in the last item of the search
    if len(moves) == 0 or depth == 0:
        # Compute the heurist function and get the current score
        return [get_heuristic_score(grid), None]

    # In this case we need to go through the different nodes.    
    if (maximize):
        # Initialize the values
        max_value = -sys.maxsize
        max_move = None
        # In this case the  nextalgorithm will be minimize
        for move in moves:
            # Clone the current grid and move accordingly
            next_grid = grid.clone()
            next_grid.move(move)
            # Add a random tile 2 or 4
            insert_random_tile(next_grid)
            # Get the score move next value
            next_value, next_move = alpha_beta(next_grid, depth-1, alpha, beta, False)
            # Compare the previous values
            if next_value > max_value:
                max_value = next_value
                max_move = move
            # Update alpha for max player
            alpha = max(max_value, alpha) 
            #Check if we must cut in this root
            if (beta <= alpha):
                break
        # Return the value
        return [max_value, max_move]
    else:
        # Initialize the value
        min_value = sys.maxsize
        min_move = None
        # In this case the  nextalgorithm will be minimize
        for move in moves:
            # Clone the current grid and move accordingly
            next_grid = grid.clone()
            next_grid.move(move)
            # Add a random tile 2 or 4
            insert_random_tile(next_grid)
            # Get to the max value
            next_value, next_move = alpha_beta(next_grid, depth-1, alpha, beta, True)
            # Compare the previous values
            if next_value < min_value:
                min_value = next_value
                min_move = move
            # Update beta for min player
            beta = min(min_value, beta) 
            #Check if we must cut in this root
            if (beta <= alpha):
                break
        #Return the value
        return [min_value, min_move]

def alpha_beta_pruning(grid, depth=10):
    """ This algorithm will go through all the availables nodes
    until it reach a leaf node. In this moment the minmax or
    alpha Beta pruning (in this case), peform an heuristic function
    to determine the final score for that movement.

    We have the grid and the availabe movements we could do.
    
    moves = grid.getAvailableMoves()
    However the movements are constrained always by for: 
        [UP, DOWN, LEFT, RIGHT] => [0,1,2,3]

    In between the maxmin players there are another step to consider,
    this is the tile that will be added per iteration betwenn them. 
    The position of this tile will be determined by the current available
    tiles and the chances to be either 2 or 4. The assignments says this
    will be automatically added in computer turn. However for our prediction
    we need to insert for our own to iterate between the tree.

    Something to consider is to optimize the algorihm so it don't
    go through all the different moves, since it will take so long 
    to compute only one node. Insteal we can perform N minmax moves
    and get the best one.

    Another problem could be the using of recursion since Python is
    not very good at performing this kind of operation.

    The Output will be the final move with better score.

    """
    # Since the game tries to maximize the score we need to start
    # by using the max search first. 
    return alpha_beta(grid, depth, -sys.maxsize, sys.maxsize, True)

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
        move = alpha_beta_pruning(grid_cloned, PlayerAI.MAX_DEPTH)
        #Return the current move performed or None if end.
        return move[1]

       

