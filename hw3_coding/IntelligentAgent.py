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
#https://en.wikipedia.org/wiki/Expectiminimax
#https://cs.nyu.edu/~fergus/teaching/ai/slides/lecture7.pdf - 3
#https://web.stanford.edu/class/archive/cs/cs221/cs221.1186/lectures/games1.pdf
#-------------------------------------------------------------------------#


#representation of movements... the directions, 
# 0 - Up, 1 - Down, 2 - Right 3 - Left
movedirs = [0,1,2,3]

class IntelligentAgent(BaseAI):
    #getMove() inherited returns int from 0 -3 for the possible moves. 
    #has and executes the player optimizing logic.
    #Move takes at most 0.2s to be computed
    max_time = 0.2
    def iterativedepth(self, grid): #following the FAQ advise that we're not to use a fixed/depth restriction, implemented the iterative deepening instead
        depth = 1
        nextMove = None
        while True:
            moveVal = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            currMove = None
            possiblemoves = grid.getAvailableMoves()
            for move, nextState in possiblemoves:
                val = self.expectiminimax(nextState, depth, alpha, beta, 1.0)
                if val > moveVal:
                    moveVal = val
                    currMove = move
                if alpha >= beta:
                    break
                if val > alpha:
                    alpha = val
            
            endtime = time.time()
            if endtime - self.starttime > 0.2:
                break
            if currMove is not None:
                nextMove = currMove
            depth = depth + 1

        return nextMove

    def getMove(self, grid):
        self.starttime = time.time()
        nextMove = self.iterativedepth(grid)
        return nextMove
    
    '''In implementing the playerAi to determine moves consider: '''
    #1. Use of expectmin-max algo - 90% of tiles are 2's and the rest are 4's
    #2. Alpha-Beta pruning to remove irrelevant branches, reducing the search space and search done**??
    #3. Use of heuristic functions - important in assigning vals to nodes and implements the time constraints, 
    # cuts off before the time constraint exceeds 0.2 s [same as eval functions for non-terminal pos]
    #4. Consider using heuristic weights to make work easier, cause might use >1 herusitic functions. 
    def expectiminimax(self, grid, depth, alpha, beta, childProb):
        if depth == 0 or not grid.getAvailableCells():
            return self.heuristicValue(grid)
        #chance logic, computer player is the minimizing agent
        chanceProb = 0
        chance = 0
        chanceval = 0
        for cell in grid.getAvailableCells():
            for prob, tileVal in ((0.9, 2), (0.1, 4)):
                tileProb = prob * childProb
                # move to next iteration if the placing of the current tile is not feasible, helps account for the uncertainity 
                if 0.9 * tileProb < 0.1 and len(grid.getAvailableCells()): 
                    continue
                gridCopy = grid.clone()
                gridCopy.setCellValue(cell, tileVal)
                chance = chance + (prob * self.maximizingLogic(gridCopy, depth, alpha, beta, tileProb))
                chanceProb = chanceProb + prob
            endtime = time.time()
            if endtime - self.starttime > 0.2:
                break
        if chanceProb == 0:
            return self.heuristicValue(grid)
        chanceval = chance/chanceProb
        return chanceval

    def maximizingLogic(self, grid, depth, alpha, beta, childProb):
        maxVal = float('-inf')
        for move, nextState in grid.getAvailableMoves():
            moveval = self.expectiminimax(nextState, depth -1, alpha, beta, childProb)
            if moveval > maxVal:
                maxVal = moveval
            if maxVal >= beta:
                break
            if maxVal > alpha:
                alpha = maxVal

        return maxVal        
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


    