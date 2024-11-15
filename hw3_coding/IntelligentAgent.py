#Working on AI homework 3 - 1024/2048  intelligent AI puzzle solver.
from BaseAI import BaseAI
import random
import time
import math

#-------------------------------------------------------------------------#
# Name:STANLEY OMONDI
# UNI: soo2117
# https://informatika.stei.itb.ac.id/~rinaldi.munir/Stmik/2013-2014-genap/Makalah2014/MakalahIF2211-2014-037.pdf
# https://theresamigler.com/wp-content/uploads/2020/03/2048.pdf
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
        nextMove = random.choice(movedirs)
        value  = float('-inf')
        time_limit = 0.2 
        starttime = time.process_time()
        for move in grid.getAvailableMoves():
            nextstate = move[1]
            moveval = self.expectminimax(nextstate, depth=2, alpha=float('-inf'), beta=float('inf'),  agent=True)
            if moveval > value:
                nextMove = move[0]
                value = moveval
                if time.process_time() - starttime > time_limit:
                    return nextMove
        return nextMove
    
    '''In implementing the playerAi to determine moves consider: '''
    #1. Use of expectmin-max algo - 90% of tiles are 2's and the rest are 4's
    #2. Alpha-Beta pruning to remove irrelevant branches, reducing the search space and search done**??
    #3. Use of heuristic functions - important in assigning vals to nodes and implements the time constraints, 
    # cuts off before the time constraint exceeds 0.2 s [same as eval functions for non-terminal pos]
    #4. Consider using heuristic weights to make work easier, cause might use >1 herusitic functions. 

    def expectminimax(self, grid, depth, alpha, beta, agent):
        if depth == 0 or not grid.getAvailableCells(): #handling the terminal nodes
            return self.heuristicValue(grid)
        
        if agent:
            bestVal = float('-inf')
            for move in grid.getAvailableMoves():
                nextgrid = move[1]
                currval = self.expectminimax(nextgrid, depth-1, alpha, beta, not agent)
                bestVal = max(bestVal, currval)
                alpha = max(alpha, bestVal)
                if beta <= alpha:
                    break
            return bestVal
        else:
            bestVal = float('inf')
            for cell in grid.getAvailableCells():
                chanceval = 0
                for prob, tileval in [(0.9 , 2), (0.1 , 4)]:
                    gridCopy = grid.clone()
                    gridCopy.setCellValue(cell, tileval)
                    currval = self.expectminimax(gridCopy, depth -1, alpha, beta, agent)
                    chanceval = chanceval + (prob * currval)
                    chanceval = chanceval / 2
                    bestVal = min(bestVal, chanceval)
                    beta = min(beta, chanceval)
                    if beta <= alpha:
                        break
            return bestVal
    

    #Working of heuristics
    def heuristicValue(self, grid):
        empty = len(grid.getAvailableCells())
        maxCellVal = max(cell for row in grid.map for cell in row)

        heursitic = math.log2(maxCellVal) + empty *2 + self.monotocity(grid) + self.smoothness(grid)
        
        return heursitic
        #calculationg the monotonistic heuristic for the grid, checks if neighbors are in increasing 
        #or decreasing order across rows and columns.
        
    def monotocity(self, grid):    
        monotonicity_score = 0
        for row in range(0, 3):
            for col in range(0, 2):
                if grid.getCellValue((row, col)) >= grid.getCellValue((row, col + 1)):
                    monotonicity_score = monotonicity_score + 1
            
        for col in range(0,3):
            for row in range(0,2):
                if grid.getCellValue((row, col)) >= grid.getCellValue((row + 1, col)):
                    monotonicity_score = monotonicity_score + 1
            
        return monotonicity_score
    

    def smoothness(self, grid):
        smoothness_score = 0
        for row in range(4):
            for col in range(4):
                curval = grid.map[row][col] 
                if col < 3:
                    nextval = grid.map[row][col + 1]
                    if nextval != 0 and curval != 0:
                        smoothness_score = smoothness_score - abs(math.log2(curval) - math.log2(nextval)) 

                if row < 3:
                    nextval = grid.map[row + 1][col]
                    if nextval != 0 and curval != 0:
                        smoothness_score = smoothness_score - abs(math.log2(curval) - math.log2(nextval)) 
 
        return smoothness_score


    