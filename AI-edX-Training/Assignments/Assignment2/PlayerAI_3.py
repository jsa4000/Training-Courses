import sys
from math import log
from random import randint
from BaseAI_3 import BaseAI

adjacent_tiles = ((-1,0),(0,1),(1,0),(0,-1))
percentage_monotonicity = [1, 0.50, 0.10, 0.0]

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

def transpose_matrix(matrix):
    """ Transpose the matrix
    """
    sizey = len(matrix[0])
    sizex = len(matrix)
    result = []
    for y in range(sizey):
        result.append([matrix[x][y] for x in range(sizex)])
    return result    

def flip_matrix(matrix, axis=0):
    """ Flip the matrix.
        Where:
            axis = 0 = Horizontal
            axis = 1 = Vertical
    """
    size = len(matrix)
    result = []
    if axis:
        for x in reversed(range(size)):
            result.append([matrix[x][y] for y in range(size)])
    else:
        for x in range(size):
            result.append([matrix[x][y] for y in reversed(range(size))])
    # Return the matrix
    return result     

def get_monotonicity_percentage_matrix():
    """ This will return the percentage matrix using the monotonicity
    defined previously. This will compute the left tiles by multiplying
    the cols and row.
    """
    result = []
    for x in percentage_monotonicity:
        result.append([x*y for y in percentage_monotonicity])
    return result
    
def create_empty_matrix(size, default_value):
    """
    """
    result = []
    for x in range(size):
        result.append([default_value for y in range(size)])
    return result

def combine_matrix(matrix1, matrix2):
    """ Combine marix2 with matrix1 where matrix1>matrix2
    if None value then the value will be replaced
    """
    sizey = len(matrix2[0])
    sizex = len(matrix2)
    for x in range(sizex):
        for y in range(sizey): 
            if matrix1[x][y] is None:
                matrix1[x][y] = matrix2[x][y]
            else:
                matrix1[x][y] = all((matrix1[x][y], matrix2[x][y]))
    return matrix1      

def get_monotonicity_matrix(matrix):
    """ Get the monotonicity of the given matrix
    """
    size = len(matrix)
    # Check from (0,0) too (3,3)
    rows = []
    # Compute the rows
    for x in range(size):
        rows.append([True if first>=second and first!=0 else False for first, second in zip(matrix[x],matrix[x][1:])])
    columns = []
    # Transpose the Matrix
    transposed = transpose_matrix(matrix)
    #Compte the columns
    for x in range(size):
        columns.append([True if first>=second and first!=0  else False for first, second in zip(transposed[x],transposed[x][1:])])
    columns = transpose_matrix(columns)
    # Combine the Matrix row and cols
    result = create_empty_matrix(size, None)
    result = combine_matrix(result,rows)
    result = combine_matrix(result,columns)
    # Return the monotonocity being true, false or None
    result[size-1][size-1] = False
    return result

def get_monotonicity_score(monotonicity_marix,monotonicity_percentage_matrix ):
    #Final score
    size = len(monotonicity_marix)
    score = 0
    # Get the score combining both matrices
    for x in range(size):
        for y in range(size):
            score += monotonicity_percentage_matrix[x][y]*monotonicity_marix[x][y]
    return score

def get_monoticity(grid):
    """ Get the monotonicity (score) for the current grid state
    """
    # Get the monocity percentages to use in the score
    monotonicity_percentage_matrix = get_monotonicity_percentage_matrix()
    scores=[]
    # This will return the monotonicity matrix from 0_0
    monotonicity_from_0_0 = get_monotonicity_matrix(grid.map)
    scores.append(get_monotonicity_score(monotonicity_from_0_0,monotonicity_percentage_matrix))
    # This will return the monotonicity matrix fomr 3_0
    monotonicity_from_3_0 = get_monotonicity_matrix(flip_matrix(grid.map, axis=0))
    scores.append(get_monotonicity_score(monotonicity_from_3_0,monotonicity_percentage_matrix))
    # This will return the monotonicity matrix fomr 0_3
    monotonicity_from_0_3 = get_monotonicity_matrix(flip_matrix(grid.map, axis=1))
    scores.append(get_monotonicity_score(monotonicity_from_0_3,monotonicity_percentage_matrix))
    # This will return the monotonicity matrix fomr 3_3
    monotonicity_from_3_3 = get_monotonicity_matrix(flip_matrix(flip_matrix(grid.map, axis=1),axis=0))
    scores.append(get_monotonicity_score(monotonicity_from_3_3,monotonicity_percentage_matrix))
    # Return the final score
    return max(scores)
  
def evaluate(grid):
    """ Get another function to evaluate the heuristic
    """
    num_blank = len(grid.getAvailableCells())
    max_tile = grid.getMaxTile()
    if num_blank == 0:
        return log(max_tile, 2) - 100
    monoticity = get_monoticity(grid) * 2
    if log(max_tile, 2) >= 8:
        monoticity = get_monoticity(grid) * 2
    #smoothness = get_smoothness(grid) * 0.1 + get_smoothness2(grid)
    smoothness = get_smoothness(grid) * 0.1
    return log(max_tile, 2) * 10 + num_blank * 3 + monoticity + smoothness * 0.1

def insert_random_tile(grid, cell=None):
    """ Insert a value [2, 4] with a probability and added  
    into a random cell available in the grid
    """
    # Get a random vale with probability = 0.9 for 2
    if randint(0,99) < 100 * 0.9:
        tileValue = 2
    else:
        tileValue = 4;
    if cell is None:
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
        return (evaluate(grid), None)
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
            # Get the move and score of the minimizer
            value, _ = alpha_beta(next_grid, depth-1, alpha, beta, False)
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
        min_value = sys.maxsize
        # Get all the available tiles
        cells = grid.getAvailableCells()
        # Go through all the moves allowed
        for cell in cells:
            #Clone current grid to insert a random tile
            next_grid = grid.clone()
            # Now lets add a random tile for computer player
            insert_random_tile(next_grid, cell)
            # Get the move and score of the maximizer
            value, _ = alpha_beta(next_grid, depth-1, alpha, beta, True)
            # Now update the max value and the alpha if changes
            if value < min_value:
                min_value =  value 
            # Check if we must to break the loop before update beta
            if min_value <= alpha: 
                break
            # Finally update beta
            beta = min(min_value, beta)
        #return the current values for the minimizer
        return (min_value, None)

class PlayerAI(BaseAI):
    
    # const tha will be used to optimize the algorithm 
    MAX_DEPTH = 3

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

       

