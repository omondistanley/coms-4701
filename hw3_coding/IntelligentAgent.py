#Working on AI homework 3 - 1024/2048  intelligent AI puzzle solver.
from BaseAI import BaseAI
import random
import time

#-------------------------------------------------------------------------#
# Name:STANLEY OMONDI
# UNI: soo2117
# https://informatika.stei.itb.ac.id/~rinaldi.munir/Stmik/2013-2014-genap/Makalah2014/MakalahIF2211-2014-037.pdf
#-------------------------------------------------------------------------#


#representation of movements... the directions, 
# 0 - Up, 1 - Down, 2 - Right 3 - Left
movedirs = [0,1,2,3]

class IntelligentAgent(BaseAI):

    #getMove() inherited returns int from 0 -3 for the possible moves. 
    #has and executes the player optimizing logic.
    #Move takes at most 0.2s to be computed
    def getMove(self, grid):
        #moveset = grid.getAvailableMoves()
        #return random.choice(moveset)[0]
        nextMove = None
        value  = float('-inf')
        for move in grid.getAvailableMoves():
            nextstate = move[1]
            moveval = self.expectminimax(nextstate, depth=2, agent=True)
            if moveval > value:
                nextMove = move[0]
                value = moveval
        return nextMove
    
    '''In implementing the playerAi to determine moves consider: '''
    #1. Use of expectmin-max algo - 90% of tiles are 2's and the rest are 4's
    #2. Alpha-Beta pruning to remove irrelevant branches, reducing the search space and search done**??
    #3. Use of heuristic functions - important in assigning vals to nodes and implements the time constraints, 
    # cuts off before the time constraint exceeds 0.2 s [same as eval functions for non-terminal pos]
    #4. Consider using heuristic weights to make work easier, cause might use >1 herusitic functions. 

    def expectminimax(self, grid, depth, agent):
        if depth == 0 or not grid.getAvailableCells(): #handling the terminal nodes
            return 0
        
        if agent:
            bestVal = float('-inf')
            for move in grid.getAvailableMoves():
                nextgrid = move[1]
                currval = self.expectminimax(nextgrid, depth-1, not agent)
                bestVal = max(bestVal, currval)
            return bestVal
        else:
            bestVal = 0
            for cell in grid.getAvailableCells():
                for prob, tileval in [(0.9 , 2), (0.1 , 4)]:
                    gridCopy = grid.clone()
                    gridCopy.setCellValue(cell, tileval)
                    currval = self.expectminimax(gridCopy, depth -1, agent)
                    bestVal = bestVal + (prob * currval)
            return bestVal

