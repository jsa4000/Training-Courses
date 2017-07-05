import os
import sys
import time
import math
import numpy as np
import pandas as pd
import copy, queue

regions = ([['c'+str(i)+str(j) for i in range(0,9)] for j in range(0,9)] + # columns - these are the forward checking sets
           [['c'+str(j)+str(i) for i in range(0,9)] for j in range(0,9)] + # rows
           [['c'+str(i)+str(j) for i in range(3*y,3*y+3) for j in range(3*x, 3*x+3)] for x in range(0,3) for y in range(0,3)]) # boxes

class csp: # object which holds the state of a board
    def __init__(self, domains, arcs, constraint):
        self.domains = domains # key -> [1-9]
        self.arcs = arcs # key -> [neighbor, neighbor, ...]
        self.constraint = constraint # function(i,j) returns true if i,j satisfy the constraint

def readSudoku(line):
    def boxRange(x): return range(int((x/3)*3),int((x/3)*3+3)) #
    def constraint(i, j): return (i != j)
    def arcgen(x,y): # generates a list of the keys which x,y constrains
        return ['c'+str(i)+str(j) for i in range(0,9) for j in range(0,9) if 
                (i != x or y != j) and (i == x or j == y or (i in boxRange(x) and j in boxRange(y)))]
    # data = [('c'+str(i)+str(j), c) for i, line in enumerate(open(file)) for j, c in enumerate(line[0:-1])]
    data =  [('c'+str(i)+str(j),line[i*8+j]) for i in range(0,9) for j in range(0,9)    ]

    domains = {key: ([range(1,10)] if c == 0 else [int(c)]) for (key, c) in data}
    arcs = {key: arcgen(int(key[1]),int(key[2])) for (key, c) in data}
    return csp(domains, arcs, constraint)

def printSudoku(csp): # print the board state nicely
    out = [range(0,9) for i in range(0,9)]
    for x in csp.domains:
        out[int(x[1])][int(x[2])] = str(csp.domains[x][0]) if len(csp.domains[x]) == 1 else 0
    for x in out: print (''.join(x))

def removeInconsistent(csp, i, j): # Standard AC3 with its helper method, based on AIMA pseudocode
    removed = False
    for x in csp.domains[i][:]:
        if not any([csp.constraint(x, y) for y in csp.domains[j]]):
            csp.domains[i].remove(x)
            removed = True
    return removed

def AC3(csp):
    queue = [(i, j) for i in csp.domains for j in csp.arcs[i]]
    while queue:
        i, j = queue.pop()
        if removeInconsistent(csp, i, j):
            for k in csp.arcs[i]:
                if i != k: queue.append((k, i))
    return csp

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
    # methods = ["BTS","AC3"]
    methods = ["AC3"]

    # Use the inputs and generate the outputs
    outputs = []
    for input in inputs:
        for method in methods:
            # Play the game of sudoku adn retur the final state
            result = AC3(readSudoku(input))
            if result:
                # If sudoku ends succesfully, append to the outputs
                outputs.append([result,method])
      
    # Write current outputs into the output file
    with open(output_file,"w") as file:
        for output in outputs:
            file.write("{}\n".format(' '.join(output)))

    # End of the Program




