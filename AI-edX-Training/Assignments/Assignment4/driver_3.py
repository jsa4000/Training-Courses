import os
import sys
import time
import math
import numpy as np
import pandas as pd

def ac3 (csp):
    ''' AC-3 algorithm

    The AC-3 algorithm (short for Arc Consistency Algorithm #3) is one of a 
    series of algorithms used for the solution of constraint satisfaction 
    problems (or CSP's). It was developed by Alan Mackworth in 1977. The 
    earlier AC algorithms are often considered too inefficient, and many 
    of the later ones are difficult to implement, and so AC-3 is the one 
    most often taught and used in very simple constraint solvers.

    This algorithm will propagate the constraints and reduce the domain 
    size of the variables by ensuring all possible (future) assignments 
    consistent

    Input: type CSP Instances.

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
                                                    there exists a relation R2(x, z) 
                                                    or a relation R2(z, x) }

            while worklist not empty
  
    '''
    # Create the queue for all the arcs,  (x, y)
    queue = [(x,y) for x in csp.binary_constraints 
                   for y in csp.binary_constraints[x]]

    # Iterate through over the queue if items
    while len(queue):
        # Dequeue an item x, y
        x, y = queue.pop()
        # Chech the values not constrained
        if arc_reduce(x, y, csp):
            if not len(csp.domains[x]):
                # Error no possible values for x
                return False
            else:
                #Append following arcs
                queue += [(z,x) for z in csp.binary_constraints
                          if z != x and z != y and x in csp.binary_constraints[z]]
    # Return True
    return True

def arc_reduce(x, y, csp):
    ''' Arc reduce method

    AC-3 proceeds by examining the arcs between pairs of 
    variables (x, y). It removes those values from the domain 
    of x which aren't consistent with the constraints between x 
    and y. The algorithm keeps a collection of arcs that are yet 
    to be checked; when the domain of a variable has any values 
    removed, all the arcs of constraints pointing to that pruned 
    variable (except the arc of the current constraint) are 
    added to the collection.

    For the particular case of Sudoku, The idea is to remove from
    the domain all the value that cannot be used because the constraints.

    Pseudo-code:
    
    function arc-reduce (x, y)
     bool change = false
     for each vx in D(x)
         find a value vy in D(y) such that vx and vy satisfy the constraint R2(x, y)
         if there is no such vy {
             D(x) := D(x) - vx
             change := true
         }
     return change

    '''
    # return change if success
    change = False

    # Loop for over all the values for the current domain
    for value_x in csp.domains[x]:
        for value_y in csp.domains[y]:
            if not csp.binary_constraints[x][y](value_x,value_y):
                # Remove value from domain
                csp.domains[x].remove(value_x)
                # Return change to True
                change = True

    # Retrun if change
    return change

class CSP:
    ''' Constraint Satisfaction Problem (CSP)

    Let's define the CSP problem. In order to solve the problem, 
    is required to define the following componentes for the CSP:

    CSP = {X, D, C} 
    Where:

    X =  {X1,X2, ... , Xn} is a set of variables,
    D =  {D1,D2, ... , Dn} is a set of the respective domains of values, and
    C =  {C1,C2, ... , Cn} is a set of constraints.

    '''
    def __init__(self, variables, domains, constraints):
        ''' Constructor
        
        The idea is to create a graph based on arcs. Each node are connected
        to contraints and these constraints connect to another node. This is
        for binary constraits win which the contraints depend on the value
        of another variable.

            NODE (V) --------> Contraint R(V,Y)  <--------------------
                                                                     |
                                                                     |
            NODE (X) --------> Contraint R(X,Y) ------->  NODE (Y)----
                |
                -------------> Contraint R(X,Z) ------->  NODE (Z)

        The idea is to create a Arc Based graph where each pair of values
        have a contraint that defined the connection or behaviour to
        be able to proceed in the search or prune.

        In the constructor all the variables will be initialized:
        variables, domains and contraints. 

        - Variables

            Variables key-value with the key as the variable and the current
            value asigned and inside the domain of the current variable. By
            default the value is None.

        - Domains

            Domain key-value in which the values of the variable could be
            assigned to. The domain will depend on the contraints and the
            type of variable.

        - Unary Contraints

            This is a list of tuples. In each tuple is defined the constraint (R). 

                [ (X, R(x1)),  (Y, R(y1)) ,  (Z, R(z1)), ..., ]

                binary_constraints[X] = R(x)
                binary_constraints[Y] = R(y)
                ...

        - Binary Contraints

            This is a list of tuples. In each tuple are defined the connection
            or arc defined by two nodes or variables, and the constraint (R). 

                [ (X, Y, R(x,y)), (X, Y, R(x,z)) , ..., ]

            The way the relationsship are going to be stored is by usign dictio-
            naries:

                binary_constraints[X][Y] = R(x,y)
                binary_constraints[X][Z] = R(x,z)
                ...
                binary_constraints[X].keys() = [Y, Z, ...] -> Output nodes

        - Binary and Unary constraints must be implement all the contraints
        inside the same constraints. The constraints will be defined by lambda
        of function expresions. 
        e.g.

            - Unary Contraint.

            (X, R(X/2 > 12)) && (X, R (X != 3))

            unary_constraints[X] = labda x: (x/2 > 12) and (x != 3)


            - Binary Contraints.

            (X, Y, R(X/2 > Y)) && (X, Y, R(X**2 > Y + 4)) && (X, Y, R(X != Y))

            binary_constraints[X][Y] = labda x,y: (x/2 > 2) and (x**2 > Y + 4) && (x != y)

        '''
        self.variables = {}
        self.domains = {}
        self.unary_constraints = {}
        self.binary_constraints = {}

        # Initialize all the variables by default.
        for index, variable in enumerate(variables):
            self.variables[variable] = None
            self.domains[variable] = domains[index]
            self.unary_constraints[variable] = None
            self.binary_constraints[variable] = {}

        # Set the constraints for each variable
        for constraint in constraints:
            if (len(constraint)>2):
                # Set the contraints for each variable, arc and constraint
                self.binary_constraints[constraint[0]][constraint[1]] = constraint[2]
            else:
                 # Set the unary contraint for each variable
                self.unary_constraints[constraint[0]] = constraint[2]
        

    def __str__(self):
        ''' Return the string representation for the CSP
        '''
        result = []
        for variable in self.variables:
            # Print Header
            result.append("Variable: {} | Value: {} | Domain: {}".format(variable,
                                                                         self.variables[variable],
                                                                         self.domains[variable]))
            result.append("|".join(list("{}=>{}".format(constraint[0],constraint[1]) \
                                    for constraint in self.binary_constraints[variable])))
            result.append("***************************************************")
        #Return the current output in separated lines
        return "\n".join(result)

class Sudoku:
    ''' Sodoku board Class

    Consider the Sudoku puzzle game. There are 81 variables (9 x 9)
    in total, i.e. the tiles to be filled with digits. Each variable is
    named by its row and its column, and must be assigned a value from 
    1 to 9, subject to the constraint that no two cells in the same row,
    column, or box may contain the same value.
   
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

    def get_name(row, column):
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
        # Create the board with empty values.
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                self.cell[Sudoku.get_name(row,column)] = Sudoku.empty_value

    def set_board(self, board):
        ''' Set the current board

        Parameters:

            board: Definition of the board ( 81 chars )
                Type: String

        '''
        # Check the board length is right
        if len(board) != (len(Sudoku.row_names) * len(Sudoku.column_names)):
            return False
        # Parse current Board and set the values
        self.cell = {}
        index = 0
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                self.cell[Sudoku.get_name(row,column)] = int(board[index])
                index += 1
        return True

    def get_board(self):
        ''' Get the current state of the game with the Board.
        '''
        board = [self.cell[Sudoku.get_name(row,column)]
                 for column in Sudoku.column_names
                 for row in Sudoku.row_names]
        return ''.join("{}".format(value) for value in board)

    def get_current_square(self, row, column):
        ''' Get the current square variables that corresponds to the 
        current row, column position
        '''
        row_square = math.ceil((Sudoku.row_names.index(row) + 1) / 3)
        col_square = math.ceil((Sudoku.column_names.index(column) + 1) / 3)
        result = [Sudoku.get_name(Sudoku.row_names[irow], Sudoku.column_names[icol])
                  for irow in range(3*(row_square-1), row_square * 3)
                  for icol in range(3*(col_square-1), col_square * 3)] 
        return result

    def empty_cell(self, row, column):
        ''' Return wethere the cell is empty or not
        '''
        return self.get_value(row,column) == Sudoku.empty_value

    def get_value(self, row, column):
        '''  Return current value of the cell given row and col
        '''
        return self.cell[Sudoku.get_name(row,column)]

    def create_csp(self):
        ''' Create the CSP that represent the current board
        '''
        board = self.get_board()

        # Main contraints to use as lamba expression
        alldiff = lambda x,y: x!=y
        # Initialize all the variables, domains and constraints
        variables = []
        domains = []
        constraints = []
        # Create the array with all the variables to be ghessed
        for row in Sudoku.row_names:
            for column in Sudoku.column_names:
                # Set current variable (nos assigned yet)
                variables.append(Sudoku.get_name(row,column))
                # Set current domain for the current variable
                domains.append(Sudoku.domain_values 
                               if self.empty_cell(row,column) 
                               else [self.get_value(row,column)])
                # Set the binary contraints
                
                # 1. Set the columns variables to create the arcs
                constraints += [(Sudoku.get_name(row,column),
                            Sudoku.get_name(row,const_column),alldiff) 
                            for const_column in Sudoku.column_names
                            if const_column != column]
                # 2. Set the rows variables to create the arcs
                constraints += [(Sudoku.get_name(row,column),
                            Sudoku.get_name(const_row,column),alldiff)
                            for const_row in Sudoku.row_names
                            if const_row != row]
            
                # 3. Set square constraints attached to this node
                square_items = self.get_current_square(row, column)
                constraints += [
                            (Sudoku.get_name(row,column),item, alldiff)
                            for item in square_items
                            if item != Sudoku.get_name(row,column)]
 
        # Create the current CSP for the current board setup
        return CSP(variables,domains,constraints)

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

        # Create the Contrained Satisfied Problem to solve the problem
        csp = self.create_csp()
        # print(self.__str__())
        # print(csp)

        if method == 'BTS':
            #Perform the Backtracking Algorithm
            pass
        elif method == 'AC3':
            #Perform AC-3 Algorithm alone
            result = ac3(csp)
            # print(self.__str__())
            # print(csp)
            # Check if returns a valid solution
            if result: 
                # Get the domains and setup the board accordingly
                print("OK")
            else:
                print("ERROR")

        # Return current state of the game after playing
        return self.get_board()

    def __str__(self):
        ''' Represent the current State of the board
        '''
        result = []
        for row in Sudoku.row_names:
            column_values = []
            for column in Sudoku.column_names:
                column_values.append(self.cell[Sudoku.get_name(row,column)])
            result.append(column_values)
        return '\n'.join("{} ".format(item) for item in result)

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
                inputs.append(line.replace("\n",""))
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


