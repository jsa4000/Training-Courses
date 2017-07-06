import sys
import re
from copy import deepcopy

#!/usr/bin/env python
#############################################################################
# Author: Victor Lu
# Files: CSP.py
# Purpose: A data structure to represent constraint satisfaction problems
#          which consists of three components: variables, domain, constraints
#



var_char = ['A','B','C','D','E','F','G','H','I']

class CSP:

    def __init__(self,filename):
        self.X = []
        self.D = []
        self.C = []
        self.init(filename)

    #########################################################################
    # init() - initialization of adding all values for variables, domains.
    # Return: void
    #
    def init(self,filename):
        for i in range(9):
            for j in range(1,10):
                var = var_char[i]+str(j)
                self.X.append(var)
                domain = set([1,2,3,4,5,6,7,8,9])
                self.D.append(domain)
        gamelist = []
        try:
            for line in open(filename):
                if(len(line)!=10):
                    print "Error: file format invalid"
                    sys.exit(1)
                gamelist = gamelist + list(line.rstrip())
        except IOError as e:
            print "Error: file cannot be opened"
            sys.exit(1)
        for i in range(len(gamelist)):
            if(re.match("\d+",gamelist[i])):
                self.D[i] = set([int(gamelist[i])])
        self.set_constraints()

    #########################################################################
    # set_constraints() - setting all variables for arc-consistency(row,col).
    # Return: void
    #
    def set_constraints(self):
        for x in self.X:
            for y in self.X:
                if((x[0] == y[0] and x[1] != y[1]) or (x[1] == y[1] and x[0] != y[0])):
                    flag = True
                    for c in self.C:
                        if(x in c and y in c):
                            flag = False
                    if(flag):
                        self.C.append(set([x,y]))

        for a in [0,3,6]:
            for b in [0,3,6]:
                self.set_cube_constraints(a,b)

    #########################################################################
    # set_cube_constraints() - setting variables for arc-consistency(cube).
    # Return: void
    #
    def set_cube_constraints(self,a,b):
        cubelist = []
        for i in range(a,a+3):
            for j in range(b,b+3):
                x = var_char[i]+str(j+1)
                cubelist.append(x)
        for x in cubelist:
            for y in cubelist:
                if(x[0] != y[0] or x[1] != y[1]):
                    flag = True
                    for c in self.C:
                        if(x in c and y in c):
                            flag = False
                    if(flag):
                        self.C.append(set([x,y]))

    #########################################################################
    # get_neighbors() - find all connected variables(row,col,cube).
    # Return: list of neighbor variables
    #
    def get_neighbors(self,x):
        index = self.X.index(x)
        row = index / 9
        col = index % 9
        neighbors = []
        for i in range(1,10):
            var_row = var_char[row]+str(i)
            var_col = var_char[i-1]+str(col+1)
            if(i != col+1):
                neighbors.append(var_row)
            if(i != row+1):
                neighbors.append(var_col)
        a = (row / 3) * 3
        b = (col / 3) * 3
        for i in range(a,a+3):
            for j in range(b,b+3):
                y = var_char[i]+str(j+1)
                if(y != x and y not in neighbors):
                    neighbors.append(y)
        return neighbors

    #########################################################################
    # is_complete() - check if assignment has complete assigned all domains.
    # Return: boolean
    #
    def is_complete(self,assignment):
        index = 0
        for d in self.D:
            if(len(d)>1 and self.X[index] not in assignment):
                return False
            index += 1
        return True

    #########################################################################
    # is_consistent() - check if selected value consistent with assignment.
    # Return: boolean
    #
    def is_consistent(self,x,v):
        neighbors = self.get_neighbors(x)
        for n in neighbors:
            d = self.D[self.X.index(n)]
            if(len(d) == 1 and v in d):
                consistent = False
        return True 

    #########################################################################
    # print_game() - pretty print the sudoku game board to standard output.
    # Return: void 
    #
    def print_game(self):
        count = 0
        for d in self.D:
            sys.stdout.write(str(d.pop()))
            count += 1
            if((count % 9)== 0):
                print ""

    #########################################################################
    # is_solved() - check if all variables' domain has been assigned.
    # Return: boolean
    #
    def is_solved(self):
        solved = True
        for d in self.D:
            if(len(d)>1):
                solved = False
        return solved

    #########################################################################
    # assign() - apply the assignment to the csp's domains.
    # Return: void
    #
    def assign(self,assignment):
        for x in assignment:
            self.D[self.X.index(x)] = set([assignment[x]])


#############################################################################
# ac3() - algorithm for checking arc-consistency in csp data structure
# Return: boolean
#
def ac3(csp):
    queue = list(csp.C)
    while (len(queue)>0):
        c = queue[0]
        queue.remove(c)
        x_i = c.pop()
        x_j = c.pop()
        if(revise(csp,x_i,x_j)):
            if((len(csp.D[csp.X.index(x_i)])==0) or (len(csp.D[csp.X.index(x_j)])==0)):
                return False
            if(len(csp.D[csp.X.index(x_i)])>1):
                neighbors = csp.get_neighbors(x_i)
                neighbors.remove(x_j)
                for x_k in neighbors:
                    queue.append(set([x_k,x_i]))
            elif(len(csp.D[csp.X.index(x_j)])>1):
                neighbors = csp.get_neighbors(x_j)
                neighbors.remove(x_i)
                for x_k in neighbors:
                    queue.append(set([x_k,x_j]))
    return True

#############################################################################
# revise() - update the domain of one variable by excluding the domain value
#            from the other variable
# Return: boolean
#
def revise(csp,x_i,x_j): #returns true iff we revise the domain of X_i
    revised = False
    d_i = csp.D[csp.X.index(x_i)]
    d_j = csp.D[csp.X.index(x_j)]
    if(len(d_i) == 1 and d_i <= d_j):
        d_j = d_j - d_i
        csp.D[csp.X.index(x_j)] = d_j
        revised = True
    elif(len(d_j) == 1 and d_j <= d_i):
        d_i = d_i - d_j
        csp.D[csp.X.index(x_i)] = d_i
        revised = True
    return revised

#############################################################################
# backtrack() - algorithm to search for solution and add to assignment
# Return: assignment or False
#
def backtrack(assignment,csp):
    if(csp.is_complete(assignment)):
        return assignment
    x = mrv(assignment,csp)
    csp_orig = deepcopy(csp)
    for v in csp.D[csp.X.index(x)]:
        inferences = {}
        if(csp.is_consistent(x,v)):
            assignment[x] = v
            inferences = forward_check(assignment,csp,x,v)
            if isinstance(inferences,dict):
                assignment.update(inferences)
                result = backtrack(assignment,csp)
                if isinstance(result,dict):
                    return result
        del assignment[x]
        if isinstance(inferences,dict):
            for i in inferences:
                del assignment[i]
        csp = deepcopy(csp_orig)
    return False

#############################################################################
# mrv() - minimum-remaining-value (MRV) heuristic
# Return: the variable from amongst those that have the fewest legal values
#
def mrv(assignment,csp):
    unassigned_x = {}
    index = 0
    for d in csp.D:
        if(len(d) > 1):
            unassigned_x[csp.X[index]] = len(d)
        index += 1
    sorted_unassigned_x = sorted(unassigned_x, key=unassigned_x.get)
    for x in sorted(unassigned_x, key=unassigned_x.get):
        if(x not in assignment):
            return x
    return False

#############################################################################
# forward() - implement Inference finding in the neighbor variables
# Return: dict of inferences
#
def forward_check(assignment,csp,x,v):
    inferences = {}
    neighbors = csp.get_neighbors(x)
    for n in neighbors:
        s = csp.D[csp.X.index(n)]
        if(len(s) > 1 and v in s):
            s = s - set([v])
            csp.D[csp.X.index(n)] = s
            if(len(s) == 1 and n not in assignment):
                inferences[n] = list(s)[0]
        inf_list = inferences.values()
        for inf in inf_list:
            if(inf_list.count(inf)>1):
                return False
    return inferences

#############################################################################
# main() - sudoku solver main process
# Return: void
#
def main(filename):
    csp = CSP(filename)
    if(ac3(csp)):
        if(csp.is_solved()):
            print "Sudoku Solved!"
            csp.print_game()
        else:
            assignment = backtrack({},csp)
            if isinstance(assignment,dict):
                csp.assign(assignment)
                print "Sudoku Solved!"
                csp.print_game()
            else:
                print "No solution exists"
    else:
        print "No solution exists"

main(sys.argv[1])
