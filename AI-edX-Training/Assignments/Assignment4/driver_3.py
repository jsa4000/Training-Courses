import os
import sys
import time
import numpy as np
import pandas as pd

def ac3 (X, D, R1=None, R2=None):
    ''' AC-3 algorithm

    The AC-3 algorithm (short for Arc Consistency Algorithm #3) is one of a 
    series of algorithms used for the solution of constraint satisfaction 
    problems (or CSP's). It was developed by Alan Mackworth in 1977. The 
    earlier AC algorithms are often considered too inefficient, and many 
    of the later ones are difficult to implement, and so AC-3 is the one 
    most often taught and used in very simple constraint solvers.

    Input:
        A set of variables X
        A set of domains D(x) for each variable x in X. D(x) 
                contains vx0, vx1... vxn, the possible values of x
        A set of unary constraints R1(x) on variable x that must be satisfied
        A set of binary constraints R2(x, y) on variables x and y 
                that must be satisfied

    Output:
        Arc consistent domains for each variable.
 
    Pseudo-code:

    function ac3 (X, D, R1, R2)
        // Initial domains are made consistent with unary constraints.
        for each x in X
            D(x) := { vx in D(x) | R1(x) }   
        // 'worklist' contains all arcs we wish to prove consistent or not.
        worklist := { (x, y) | there exists a relation R2(x, y)
                        or a relation R2(y, x) }
    
        do
            select any arc (x, y) from worklist
            worklist := worklist - (x, y)
            if arc-reduce (x, y) 
                if D(x) is empty
                    return failure
                else
                    worklist := worklist + { (z, x) | z != y and 
                        there exists a relation R2(x, z) or a relation R2(z, x) }
        while worklist not empty
  
    '''
    # Initialize the result
    result = []

    # Return all domains founded for each variable.
    return result

def arc-reduce(x, y):
    '''

    Pseudo-code:

    function arc-reduce (x, y)
        bool change = false
        for each vx in D(x)
            find a value vy in D(y) such that vx 
                and vy satisfy the constraint R2(x, y)
            if there is no such vy {
                D(x) := D(x) - vx
                change := true
            }
        return change

    '''
    # Initizlize the variable
    change = False

    # Retrun if change
    return change

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

        Parameters:

            board: Definition of the board ( 81 chars )
                Type: String

        '''
        # Check the board length is rgiht
        if len(board) != (len(Sudoku.row_names) * len(Sudoku.column_names)):
            return False
        # Parse current Board and set the values
        self.cell = {}
        index = 0
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                self.cell[Sudoku.get_index(row,column)] = board[index]
                index += 1
        return True

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
        if not self.set_board(board):
            return None

        if method == 'BTS':
            #Perform the Backtracking Algorithm


            pass
        elif method == 'AC3':
            #Perform AC-3 Algorithm alone
            result = ac3()
            # Check if returns a valid solution
            if not result: return None

        # Return current state of the game after playing
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
            # Play the game of sudoku adn retur the final state
            result = Sudoku().play(input,method=method)
            if result:
                # If sudoku ends succesfully, append to the outputs
                outputs.append([result,method])
      
    # Write current outputs into the output file
    with open(output_file,"w") as file:
        for output in outputs:
            file.write("{}\n".format(' '.join(output)))

    # End of the Program


