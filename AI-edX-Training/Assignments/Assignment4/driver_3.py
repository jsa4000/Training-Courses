import os
import sys
import time
import numpy as np
import pandas as pd


class Sudoku:
    ''' Sodoku board Class

    Consider the Sudoku puzzle game. There are 81 variables (9 x 9)
    in total, i.e. the tiles to be filled with digits. Each variable is
    named by its row and its column, and must be assigned a value from 
    1 to 9, subject to the constraint that no two cells in the same row,
    column, or box may contain the same value.

    Constraint Satisfaction Problem (CSP)

    Let's define the CSP problem. In order to solve the problem, 
    is required to define the following componentes for the CSP:

    CSP = {X, D, C} 
    Where:

    X =  {X1,X2, ... , Xn} is a set of variables,
    D =  {D1,D2, ... , Dn} is a set of the respective domains of values, and
    C =  {C1,C2, ... , Cn} is a set of constraints.

    SUDOKU Definition Problem using CSP

    - Variables (cell): 
    
        9 x 9 (row x columns) = 81 cells

        rows_names    = [A,B,C,D,E,F,G,H,I]
        columns_names = [1,2,3,4,5,6,7,8,9] 

        Cells [row, columns]. From A1 to I9

            1   2   3   4   5   6   7   8   9
        A  A1  A2  A3  A4  A5  A6  A7  A8  A9
        B  B1  B2  B3  B4  B5  B6  B7  B8  B9
        C  C1  ...
        D  
        E  
        F  
        G  
        H                 ...  H6  H7  H8  H9
        I  I1  I2  I3  I4  I5  I6  I7  I8  I9

    - Domains (possible values for the variables):

        All variables (cells) belong the same domain

        domain_values = [1,2,3,4,5,6,7,8,9]

        e.g. cell['A1'] = 3 , cell['I1'] = 9, ...
     
    - Constraints:

        Contrainst are the Sudoku's game rules.

        alldiff(rows)    => alldiff(A1, A9), alldiff(B1, B9) .. alldiff(I1 ,I9)
        alldiff(columns) => alldiff(A1, I1), alldiff(A2, I2) .. alldiff(A9, I9)
        adddif (squares) => alldiff(A1, C3), alldiff(A4, C6), alldiff(A7, C9)  
                            alldiff(D1, F3), alldiff(D4, F6), alldiff(D7, F9)
                            alldiff(G1, I3), alldiff(G4, I6), alldiff(G7, I9)


    '''

    # Define the ampy value for a cell (variable)
    empty_value = 0

    # Define the name of the columns and rows
    row_names    = ['A','B','C','D','E','F','G','H','I']
    column_names = ['1','2','3','4','5','6','7','8','9'] 

    # Values of range(1,10)
    domain_values = [1,2,3,4,5,6,7,8,9]

    def get_index(row, column):
        ''' Static function returning the string value joinning the
        given name and columns name. The returned value could be 
        used for Hashing the row-col name into a dictionary.
        '''
        return "{}{}".format(row,column)

    def __init__(self):
        ''' Create the Board with the variables (cells) with
        empty values. 
        '''
        self.cell = {}
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                self.cell[Sudoku.get_index(row,column)] = Sudoku.empty_value

    def set_board(self, board):
        ''' Set the current board
        '''
        self.cell = {}
        index = 0
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                self.cell[Sudoku.get_index(row,column)] = board[index]
                index += 1

    def get_board(self):
        ''' Get the current state of the game with the Board.
        '''
        board = []
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                board.append(self.cell[Sudoku.get_index(row,column)])
        return ''.join("{}".format(value) for value in board)

    def play(self, board, method='BTS'):
        ''' Play current Board game

        Parameters:
            board: Definition of the board ( 81 chars )
                Type: String

            method: 
                Type: string 
                Options:  'AC3': AC-3 Algorithm
                          'BTS': Backtracking Algorithm 

        '''
        # Set current board
        self.set_board(board)

        # REturn current state of the game after playing
        return self.get_board()


if __name__ == "__main__":
    # Define the output file
    output_file = 'output.txt'
    # Boards to be computed
    inputs = []
    # Get the parameters from Args
    if len(sys.argv)<2:
        # If not parameter the define a default string
        with open('sudokus_start.txt','r') as file:
            for line in file:
                inputs.append(line)
    else:
        # Only use the current algorithm
        inputs.append(sys.argv[1])

    # Methods availabile to solve the Sudoku game
    methods = ["BTS","AC3"]

    # Use the inputs and generate the outputs
    outputs = []
    for input in inputs:
        for method in methods:
            # PLay the game of sudoku adn retur the final state
            outputs.append([Sudoku().play(input,method=method),method])
      
    # Write current outputs into the output file
    with open(output_file,"w") as file:
        for output in outputs:
            file.write("{}\n".format(' '.join(output)))

    # End of the Program


