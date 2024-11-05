#Working on AI homework 3 - 1024/2048  intelligent AI puzzle solver.
from BaseAI import BaseAI
import time

#-------------------------------------------------------------------------#
# Name:STANLEY OMONDI
# UNI: soo2117
#-------------------------------------------------------------------------#

class IntelligentAgent(BaseAI):

    #getMove() inherited returns int from 0 -3 for the possible moves. 
    #has and executes the player optimizing logic.
    #Move takes at most 0.2s to be computed
    
    '''In implementing the playerAi to determine moves consider: '''
    #1. Use of expectmin-max algo - 90% of tiles are 2's and the rest are 4's
    #2. Alpha-Beta pruning to remove irrelevant branches, reducing the search space and search done**??
    #3. Use of heuristic functions - important in assigning vals to nodes and implements the time constraints, 
    # cuts off before the time constraint exceeds 0.2 s
    #4. Consider using heuristic weights to make work easier, cause might use >1 herusitic functions. 

