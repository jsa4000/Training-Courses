import sys
import math
from random import randint
from BaseAI_3 import BaseAI
import queue


def get_monoticity(grid):
    max_value = grid.getMaxTile()
    max_pos = (1,1)
    max_dec = -sys.maxsize
    for i in range(0, grid.size):
        for j in range(0, grid.size):
            if grid.map[i][j] == max_value:
                dec = abs(i - 1.5) + abs(j - 1.5)
                if dec > max_dec:
                    max_pos = (i, j)
                    max_dec = dec

    sum_mono = 0
    if max_dec == 3:

        for i in range(1,grid.size):
            if not grid.crossBound((max_pos[0] + i, max_pos[1])):
                if grid.getCellValue((max_pos[0] + i, max_pos[1])) == 0:
                    break
                if math.log(grid.getCellValue((max_pos[0] + i, max_pos[1])),2) - math.log(grid.getCellValue((max_pos[0] + i - 1, max_pos[1])),2) <= 1:
                    sum_mono += 1


        for i in range(1,grid.size):
            if not grid.crossBound((max_pos[0] - i, max_pos[1])):
                if grid.getCellValue((max_pos[0] - i, max_pos[1])) == 0:
                    break
                if math.log(grid.getCellValue((max_pos[0] - i, max_pos[1])),2) - math.log(grid.getCellValue((max_pos[0] - i + 1, max_pos[1])),2) <= 1:
                    sum_mono += 1


        for i in range(1,grid.size):
            if not grid.crossBound((max_pos[0], max_pos[1] + i)):
                if grid.getCellValue((max_pos[0], max_pos[1] + i)) == 0:
                    break
                if math.log(grid.getCellValue((max_pos[0], max_pos[1] + i)),2) - math.log(grid.getCellValue((max_pos[0], max_pos[1] + i - 1)),2) <= 1:
                    sum_mono += 1
                    
                    
        for i in range(1,grid.size):
            if not grid.crossBound((max_pos[0], max_pos[1] - i)):
                if grid.getCellValue((max_pos[0], max_pos[1] - i)) == 0:
                    break
                if math.log(grid.getCellValue((max_pos[0], max_pos[1] - i)),2) - math.log(grid.getCellValue((max_pos[0], max_pos[1] - i + 1)),2) <= 1:
                    sum_mono += 1
    
    if sum_mono >=3:
        return 3
    return sum_mono

def get_monoticity2(grid):
    
    max_value = grid.getMaxTile()
    max_pos = [0,0]
    max_dec = -sys.maxsize
    for i in range(0, grid.size):
        for j in range(0, grid.size):
            if grid.map[i][j] == max_value:
                dec = abs(i - 1.5) + abs(j - 1.5)
                if dec > max_dec:
                    max_pos = (i, j)
                    max_dec = dec

    sum_mono = 0

    if max_dec == 3:
        sum_mono += math.log(max_value)
        x_dir = 0
        y_dir = 0
        if max_pos[0] == 0 and max_pos[1] == 0:
            x_dir = 1
            y_dir = 1
        elif max_pos[0] == 3 and max_pos[1]== 3:
            x_dir = -1
            y_dir = -1
        elif max_pos[0] == 3 and max_pos[1] == 0:
            x_dir = -1
            y_dir = 1
        else:
            x_dir = 1
            y_dir = -1

        xy_dir = (x_dir, y_dir)

        sum_mono1 = 0

        cursor = deepcopy(max_pos)
        for i in range(1, grid.size * 2):
            prev = deepcopy(cursor)
            if grid.getCellValue(cursor) == 0:
                break
            if i % 4 == 0:
                cursor = (cursor[0], cursor[1] + y_dir)
                x_dir = -1 * x_dir
            else:
                cursor = (cursor[0] + x_dir, cursor[1])
            if grid.getCellValue(cursor) - grid.getCellValue(prev) >= 0 and (math.log(grid.getCellValue(cursor), 2) - math.log( grid.getCellValue(prev), 2)) <= 2:
                sum_mono1 +=  math.log( grid.getCellValue(prev), 2)
                if math.log(grid.getCellValue(cursor), 2) - math.log( grid.getCellValue(prev), 2) <= 1:
                    pass
            else:
                break

        sum_mono2 = 0

        x_dir = xy_dir[0]
        y_dir = xy_dir[1]

        cursor = deepcopy(max_pos)
        history = []
        for i in range(1, grid.size * 2):
            history.append((cursor,i))  
            prev = deepcopy(cursor)
            if grid.getCellValue(cursor) == 0:
                break
            if i % 4 == 0:
                cursor = (cursor[0] + x_dir, cursor[1])
                y_dir = -1 * y_dir
            else:
                cursor = (cursor[0], cursor[1] + y_dir)
            if grid.getCellValue(cursor) - grid.getCellValue(prev) >= 0:
                sum_mono2 += math.log( grid.getCellValue(prev), 2)
                if math.log(grid.getCellValue(cursor), 2) - math.log( grid.getCellValue(prev), 2) <= 1:
                    pass
            else:
                break

        sum_mono = sum_mono + max(sum_mono1, sum_mono2)

    return sum_mono


def get_smoothness(grid):
    tiles = []
    for i in range(0, grid.size):
        for j in range(0, grid.size):
            if grid.map[i][j] > 0:
                tiles.append((i, j))
    smoothness = 0

    for i, t in enumerate(tiles):
        for j in range(i+1, len(tiles)):
            if math.log(grid.getCellValue(t), 2) - math.log(grid.getCellValue(tiles[j]), 2) == 0 and math.log(grid.getCellValue(t), 2):
                if abs(t[0] - tiles[j][0]) + abs(t[1] - tiles[j][1]) == 1:
                    smoothness += 1 * math.log(grid.getCellValue(t), 2)/5
                    continue

    for i, t in enumerate(tiles):
        for j in range(i+1, len(tiles)):
            if math.log(grid.getCellValue(t), 2) - math.log(grid.getCellValue(tiles[j]), 2) == 1 and math.log(grid.getCellValue(t), 2):
                if abs(t[0] - tiles[j][0]) + abs(t[1] - tiles[j][1]) == 1:
                    smoothness += 0.5 * max(math.log(grid.getCellValue(t), 2), math.log(grid.getCellValue(tiles[j]), 2))/5
                    continue

    return smoothness

def get_smoothness2(grid):
    mqueue = queue.PriorityQueue(100)
    for i in range(0, grid.size):
        for j in range(0, grid.size):
            if grid.map[i][j] > 0:
                mqueue.put((20 - grid.map[i][j], (i, j)))
    tiles = []
    smoothness = 0
    while not mqueue.empty():
        tiles.append(mqueue.get()[1])

    for i in range(1, min(len(tiles), 8)):
        if abs(tiles[i-1][0] - tiles[i][0]) + abs(tiles[i][1] - tiles[i-1][1]) == 1:
            smoothness += math.log(grid.getCellValue(tiles[i]), 2)
    return smoothness

def evaluate(grid):
    """ Get another function to evaluate the heuristic
    """
    num_blank = len(grid.getAvailableCells())
    max_tile = grid.getMaxTile()
    if num_blank == 0:
        return math.log(max_tile, 2) - 100
    monoticity = get_monoticity(grid) * 2
    if math.log(max_tile, 2) >= 8:
        monoticity = get_monoticity(grid) * 2
    smoothness = get_smoothness(grid) * 0.1 + get_smoothness2(grid)
    #smoothness = get_smoothness(grid) * 0.1
    return math.log(max_tile, 2) * 10 + num_blank * 3 + monoticity + smoothness * 0.1

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

       

