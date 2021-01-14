import numpy as np
import tkinter as tk
import pandas as pd 
import scipy as sp 
import matplotlib.pyplot as plt 
import keyboard
import sys
import os
import re
import random
sys.path.append ('.')
sys.path.append("..")

from lib.const import *
from lib.utils import *

from lib.net import *

import time
from random import choice, shuffle
from math import log, sqrt

# 2020/12/5 Limzh -- Add
# Player is the class which defines agents' behaviors
class BasicAgent (object):
    def __init__ (self, piece, gameState, name:str = 'BasicPlayer'):
        self.name = name
        self.piece = piece
        self.gameState = gameState
    def getDecision (self, gameState:object):
        # Code (TBA)
        pass
    #! WARNING: `putPiece` is the interface of AI mode. YOU SHOULD NOT MODIFY IT IN ANY CASE!
    def putPiece (self, gameState:object, Xpos:int, Ypos:int, isGUI:bool = True)->int:
        '''piece should be type of Enum piece. This action is valid iff (Xpos, Ypos) is None.
        `putPiece` function is the most important function. It is the core API for AI mode.
        Basically, by passing by current game state with coordX and coordY, it invokes corresponding gameState functions -- Judge logic function \ Update function \ GUI function to maintain chessBoard status.'''
        assert (0 <= Xpos <= GRID_NUM)
        assert (0 <= Ypos <= GRID_NUM)
        assert (self.piece == Piece.BLACK or self.piece == Piece.WHITE)
        grids = gameState.getGrids ()
        if (grids[(Xpos, Ypos)] is not None):
            Helper.WARNING (f"{(Xpos, Ypos)} already has a piece.")
            return False
        else:
            grids[(Xpos, Ypos)] = self.piece
            gameState.count += 1
            if (isGUI): 
                gameState.chessBoard.drawChess (self.piece, Xpos, Ypos, "chess"+str(gameState.count))
                gameState.window.root.update ()
            gameState.currPiece = self.changePiece ()
            isWin = gameState.check (self.piece, Xpos, Ypos)
            # return reward
            if (isWin):
                gameState.winPiece = self.piece
                # print (Colored.cyan(DEFAULT_WIN_REFIX + f"{self.name} with {self.piece.value} piece wins!"))
            return True
        Helper.NOT_REACHED ()

    def changePiece (self):
        if (self.piece == Piece.BLACK): return Piece.WHITE
        else: return Piece.BLACK
        Helper.NOT_REACHED ()

    def requestRegretChess (self):
        # Code (TBA). Advanced action.
        pass
    def requestDraw (self):
        # Code (TBA). Advanced action
        pass
# 2020/12/25 Limzh -- Add
# -- please pay attention to:
# (1) Code Specification -- e.g. indentation, notation and so on
# (2) Debug Helper Usage -- e.g. PANIC, WARNING and so on
# (3) Efficiency -- If you find it low-efficient, please considering pruning methods \ pallel programming.
# (4) PLEASE, PLEASE use `Pickle` to store the param module! and name it in an explicit and clear way.
# (5) Version Control is important, please dont merge to `main` branch in a simple and brute way.
# TODO: YOU SHOULD ACCOMPLISH THE FOLLOWING AGENTS
class HumanAgent (BasicAgent):
    '''Nothing fancy, nothing more.'''
    typeName = "HUMAN"
    def __init__(self, piece, gameState, name:str='BasicPlayer'):
        super().__init__(piece, gameState, name=name)
    def getClickPos (self, event) -> bool:
        if self.gameState.currPiece != self.piece: return
        startX, startY = DRAW_LEFT_UPON
        Xpos, Ypos = round((event.x - 100)/40), round((event.y - 100)/40)
        if not (0 <= Xpos <=14 and 0 <= Ypos <= 14): 
            Helper.WARNING ("Invalid Position!")
            return False
        else:
            Xpos, Ypos = self.getDecision (Xpos, Ypos)
            return self.putPiece (self.gameState, Xpos, Ypos, self.gameState.isGUI)
            
        Helper.NOT_REACHED ()

    def getDecision (self, Xpos, Ypos):
        return (Xpos, Ypos)
        
class RandomAgent (BasicAgent):
    '''Random Agent'''
    typeName = "RandomAgent"
    def __init__ (self, piece, gameState, isGUI = True, name:str = "RandomAgent"):
        super().__init__(piece, gameState, name=name)
        self.isGUI = isGUI
    
    def getSuccessors (self):
        return self.gameState.getAvailableGrids ()
        Helper.NOT_REACHED ()

    def getDecision (self) -> tuple:
        '''Random Agent randomly pick one available.'''
        import random
        successors = self.getSuccessors ()
        return random.choice (successors)[0]
        Helper.NOT_REACHED ()
    
    def takeAction (self):
        '''putPiece'''
        Xpos, Ypos = self.getDecision ()
        self.putPiece(self.gameState, Xpos, Ypos, self.isGUI)
        return None
        Helper.NOT_REACHED ()  

class Q_Agent (BasicAgent):
    '''Q-learning Agent'''
    typeName = "QAgent"
    def __init__(self, piece, gameState, name:str='Q_Agent', isGUI = True, isTraining = False, numTraining = 3000, epsilon = 0.5, alpha = 0.01, gamma = 0.9):
        super().__init__(piece, gameState, name=name)
        
        self.isGUI = isGUI
        self.isTraining = isTraining
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.episodesSoFar = 0

        self.QValues = Counter() # Initialize QValues

    def getNumTraining(self):
        '''
          Get the total number of training
        '''
        return self.numTraining
        Helper.NOT_REACHED ()

    def getLegalActions(self):
        '''
          Get the actions available for a given
          state. 
        '''
        availableGrids = self.gameState.getAvailableGrids ()
        legalActions = [x[0] for x in availableGrids]
        return legalActions
        Helper.NOT_REACHED ()

    def getQValue(self, state, action):
        '''
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise 
        '''
        return self.QValues[state, action]
        Helper.NOT_REACHED ()

    def computeValueFromQValues(self, state):
        '''
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, it should return a value of 0.0.
        '''
        allActions = self.getLegalActions()
        if len(allActions) == 0:
            return 0.0
        maxQValue = -99999999
        for action in allActions:
            newQValue = self.getQValue(state, action)
            if newQValue > maxQValue:
                maxQValue = newQValue
        return maxQValue
        Helper.NOT_REACHED ()
        
    def computeActionFromQValues(self, state):
        '''
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        '''
        allActions = self.getLegalActions()
        if len(allActions) == 0:
            return None
        maxQValue = -99999999
        maxActions = []
        for action in allActions:
            newQValue = self.getQValue(state, action)
            if newQValue > maxQValue:
                maxActions.clear()
                maxQValue = newQValue
                maxActions.append(action)
            elif newQValue == maxQValue:
                maxActions.append(action)
        if len(maxActions) == 0:
            print(state, self.gameState.getGrids(), allActions)
        return random.choice(maxActions)
        # return maxActions[0]
        Helper.NOT_REACHED ()

    def update(self, state, action, nextState, reward):
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.QValues[state, action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getDecision(self):
        '''
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        '''
        state = self.gameState.getGridAsNumpyList()
        legalActions = self.getLegalActions()
        action = None
        explore = flipCoin(self.epsilon)
        if explore:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action
        Helper.NOT_REACHED ()

    def getValue(self, state):
        '''
          Get the value of the state, 
          which is Returns max_action Q(state,action)
        '''
        return self.computeValueFromQValues(state)
        Helper.NOT_REACHED ()

    def takeAction(self, action):
        '''
          Put the piece on the board
        '''
        Xpos, Ypos = action[0], action[1]
        self.putPiece(self.gameState, Xpos, Ypos, self.isGUI, self.isTraining)
        return None
        Helper.NOT_REACHED ()

class Approximate_Q_Agent(Q_Agent):
    '''Approximate-Q-learning Agent'''
    typeName = "ApproximateQAgent"
    def __init__(self, piece, gameState, name:str='Approximate_Q_Agent', isGUI = True, isTraining = False, numTraining = 10, epsilon = 0.5, alpha = 0.01, gamma = 0.9):
        super().__init__(piece, gameState, name, isGUI, isTraining, numTraining, epsilon, alpha, gamma)
        self.weights = Counter()

    def getWeights(self):
        return self.weights
        Helper.NOT_REACHED()

    def getFeatures(self, state, action):
        ''' 14 features in total(2 sides): (1, 0, 2), (1, 0, 3), (1, 0, 4), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 5),
                                           (-1, 0, 2), (-1, 0, 3), (-1, 0, 4), (-1, 1, 2), (-1, 1, 3), (-1, 1, 4), (-1, 5)'''
        if state[action] != None:
            print("Invalid Action!!!")
        state[action] = self.piece
        ret = Counter()
        my_piece = self.piece
        if my_piece == Piece.BLACK:
            op_piece = Piece.WHITE
        else:
            op_piece = Piece.BLACK
        # First check each row
        for row in range(GRID_NUM + 1):
            my = 0
            op = 0
            for x in range(GRID_NUM + 1):
                if (state[(x, row)] == my_piece): 
                    my += 1
                    if 2 <= my < 5:
                        # middle of the row
                        if x - my >= 0 and x + 1 <= GRID_NUM:
                            # (1, 0, my)
                            if state[(x - my, row)] == None and state[(x + 1, row)] == None:
                                ret[(1, 0, my)] += 1
                            # (1, 1, my)
                            elif (state[(x - my, row)] == op_piece and state[(x + 1, row)] == None) or (state[(x - my, row)] == None and state[(x + 1, row)] == op_piece):
                                ret[(1, 1, my)] += 1
                        # left end of the row        
                        elif x - my < 0 and x + 1 <= GRID_NUM:
                            # (1, 1, my)
                            if state[(x + 1, row)] == None:
                                ret[(1, 1, my)] += 1
                        # right end of the row
                        elif x - my >= 0 and x + 1 > GRID_NUM:
                            # (1, 1, my)
                            if state[(x - my, row)] == None:
                                ret[(1, 1, my)] += 1
                    elif my == 5:
                        # win state
                        ret[(1, 5)] += 1
                else: 
                    my = 0
                
                if (state[(x, row)] == op_piece): 
                    op += 1
                    if 2 <= op < 5:
                        # middle of the row
                        if x - op >= 0 and x + 1 <= GRID_NUM:
                            # (-1, 0, op)
                            if state[(x - op, row)] == None and state[(x + 1, row)] == None:
                                ret[(-1, 0, op)] += 1
                            # (-1, 1, op)
                            elif (state[(x - op, row)] == my_piece and state[(x + 1, row)] == None) or (state[(x - op, row)] == None and state[(x + 1, row)] == my_piece):
                                ret[(-1, 1, op)] += 1
                        # left end of the row        
                        elif x - op < 0 and x + 1 <= GRID_NUM:
                            # (-1, 1, op)
                            if state[(x + 1, row)] == None:
                                ret[(-1, 1, op)] += 1
                        # right end of the row
                        elif x - op >= 0 and x + 1 > GRID_NUM:
                            # (-1, 1, op)
                            if state[(x - op, row)] == None:
                                ret[(-1, 1, op)] += 1
                    elif op == 5:
                        # lose state
                        ret[(-1, 5)] += 1
                else: op = 0

        # Second check each column
        for col in range(GRID_NUM + 1):
            my = 0
            op = 0
            for y in range(GRID_NUM + 1):
                if (state[(col, y)] == my_piece): 
                    my += 1
                    if 2 <= my < 5:
                        # middle of the col
                        if y - my >= 0 and y + 1 <= GRID_NUM:
                            # (1, 0, my)
                            if state[(col, y - my)] == None and state[(col, y + 1)] == None:
                                ret[(1, 0, my)] += 1
                            # (1, 1, my)
                            elif (state[(col, y - my)] == op_piece and state[(col, y + 1)] == None) or (state[(col, y - my)] == None and state[(col, y + 1)] == op_piece):
                                ret[(1, 1, my)] += 1
                        # top of the column       
                        elif y - my < 0 and y + 1 <= GRID_NUM:
                            # (1, 1, my)
                            if state[(col, y + 1)] == None:
                                ret[(1, 1, my)] += 1
                        # bottom of the column
                        elif y - my >= 0 and y + 1 > GRID_NUM:
                            # (1, 1, my)
                            if state[(col, y - my)] == None:
                                ret[(1, 1, my)] += 1
                    elif my == 5:
                        # win state
                        ret[(1, 5)] += 1
                else: 
                    my = 0
                
                if (state[(col, y)] == op_piece): 
                    op += 1
                    if 2 <= op < 5:
                        # middle of the col
                        if y - op >= 0 and y + 1 <= GRID_NUM:
                            # (-1, 0, op)
                            if state[(col, y - op)] == None and state[(col, y + 1)] == None:
                                ret[(-1, 0, op)] += 1
                            # (-1, 1, op)
                            elif (state[(col, y - op)] == my_piece and state[(col, y + 1)] == None) or (state[(col, y - op)] == None and state[(col, y + 1)] == my_piece):
                                ret[(-1, 1, op)] += 1
                        # top of the column      
                        elif y - op < 0 and y + 1 <= GRID_NUM:
                            # (-1, 1, op)
                            if state[(col, y + 1)] == None:
                                ret[(-1, 1, op)] += 1
                        # bottom of the column
                        elif y - op >= 0 and y + 1 > GRID_NUM:
                            # (-1, 1, op)
                            if state[(col, y - op)] == None:
                                ret[(-1, 1, op)] += 1
                    elif op == 5:
                        # lose state
                        ret[(-1, 5)] += 1
                else: op = 0

        # Third check each diagonal(left start)
        for row_ in range(GRID_NUM, -1, -1):
            row = row_
            my = 0
            op = 0
            for col in range(GRID_NUM - row + 1):
                if (state[(col, row)] == my_piece): 
                    my += 1
                    if 2 <= my < 5:
                        # middle of the diagonal
                        if col - my >= 0 and row - my >= 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:
                            # (1, 0, my)
                            if state[(col - my, row - my)] == None and state[(col + 1, row + 1)] == None:
                                ret[(1, 0, my)] += 1
                            # (1, 1, my)
                            elif (state[(col - my, row - my)] == op_piece and state[(col + 1, row + 1)] == None) or (state[(col - my, row - my)] == None and state[(col + 1, row + 1)] == op_piece):
                                ret[(1, 1, my)] += 1
                        # top-left of the diagonal    
                        elif (col - my < 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM):   
                            # (1, 1, my)
                            if state[(col + 1, row + 1)] == None:
                                ret[(1, 1, my)] += 1
                        # bottom-right of the diagonal
                        elif (col - my >= 0 and row - my >= 0 and row + 1 > GRID_NUM):
                            # (1, 1, my)
                            if state[(col - my, row - my)] == None:
                                ret[(1, 1, my)] += 1
                    elif my == 5:
                        # win state
                        ret[(1, 5)] += 1
                else: 
                    my = 0
                
                if (state[(col, row)] == op_piece): 
                    op += 1
                    if 2 <= op < 5:
                        # middle of the diagonal
                        if col - op >= 0 and row - op >= 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:
                            # (-1, 0, op)
                            if state[(col - op, row - op)] == None and state[(col + 1, row + 1)] == None:
                                ret[(-1, 0, op)] += 1
                            # (-1, 1, op)
                            elif (state[(col - op, row - op)] == my_piece and state[(col + 1, row + 1)] == None) or (state[(col - op, row - op)] == None and state[(col + 1, row + 1)] == my_piece):
                                ret[(-1, 1, op)] += 1
                        # top-left of the diagonal    
                        elif (col - op < 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM):   
                            # (-1, 1, op)
                            if state[(col + 1, row + 1)] == None:
                                ret[(-1, 1, op)] += 1
                        # bottom-right of the diagonal
                        elif (col - op >= 0 and row - op >= 0 and row + 1 > GRID_NUM):
                            # (-1, 1, op)
                            if state[(col - op, row - op)] == None:
                                ret[(-1, 1, op)] += 1
                    elif op_piece == 5:
                        # lose state
                        ret[(-1, 5)] += 1
                else: 
                    op = 0
                if row + 1 <= GRID_NUM:
                    row += 1
        # top start
        for col_ in range(1, GRID_NUM + 1):
            col = col_
            my = 0
            op = 0
            for row in range(GRID_NUM - col_ + 1):
                if (state[(col, row)] == my_piece): 
                    my += 1
                    if 2 <= my < 5:
                        # middle of the diagonal
                        if col - my >= 0 and row - my >= 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:
                            # (1, 0, my)
                            if state[(col - my, row - my)] == None and state[(col + 1, row + 1)] == None:
                                ret[(1, 0, my)] += 1
                            # (1, 1, my)
                            elif (state[(col - my, row - my)] == op_piece and state[(col + 1, row + 1)] == None) or (state[(col - my, row - my)] == None and state[(col + 1, row + 1)] == op_piece):
                                ret[(1, 1, my)] += 1
                        # top-left of the diagonal    
                        elif col - my >= 0 and row - my < 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:   
                            # (1, 1, my)
                            if state[(col + 1, row + 1)] == None:
                                ret[(1, 1, my)] += 1
                        # bottom-right of the diagonal
                        elif col - my >= 0 and row - my >= 0 and col + 1 > GRID_NUM and row + 1 <= GRID_NUM:
                            # (1, 1, my)
                            if state[(col - my, row - my)] == None:
                                ret[(1, 1, my)] += 1
                    elif my == 5:
                        # win state
                        ret[(1, 5)] += 1
                else: 
                    my = 0

                if (state[(col, row)] == op_piece): 
                    op += 1
                    if 2 <= op < 5:
                        # middle of the diagonal
                        if col - op >= 0 and row - op >= 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:
                            # (-1, 0, op)
                            if state[(col - op, row - op)] == None and state[(col + 1, row + 1)] == None:
                                ret[(-1, 0, op)] += 1
                            # (-1, 1, op)
                            elif (state[(col - op, row - op)] == my_piece and state[(col + 1, row + 1)] == None) or (state[(col - op, row - op)] == None and state[(col + 1, row + 1)] == my_piece):
                                ret[(-1, 1, op)] += 1
                        # top-left of the diagonal    
                        elif col - op >= 0 and row - op < 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:   
                            # (-1, 1, op)
                            if state[(col + 1, row + 1)] == None:
                                ret[(-1, 1, op)] += 1
                        # bottom-right of the diagonal
                        elif col - op >= 0 and row - op >= 0 and col + 1 > GRID_NUM and row + 1 <= GRID_NUM:
                            # (-1, 1, op)
                            if state[(col - op, row - op)] == None:
                                ret[(-1, 1, op)] += 1
                    elif op_piece == 5:
                        # lose state
                        ret[(-1, 5)] += 1
                else: 
                    op = 0
                if col + 1 <= GRID_NUM:
                    col += 1
            
        # Finally check each counter-diagonal(right start)
        for row_ in range(GRID_NUM + 1):
            row = row_
            my = 0
            op = 0
            for col in range(GRID_NUM, row_ - 1, -1):
                if (state[(col, row)] == my_piece): 
                    my += 1
                    if 2 <= my < 5:
                        # middle of the counter-diagonal
                        if col + my <= GRID_NUM and row - my >= 0 and col - 1 >= 0 and row + 1 <= GRID_NUM:
                            # (1, 0, my)
                            if state[(col + my, row - my)] == None and state[(col - 1, row + 1)] == None:
                                ret[(1, 0, my)] += 1
                            # (1, 1, my)
                            elif (state[(col + my, row - my)] == op_piece and state[(col - 1, row + 1)] == None) or (state[(col + my, row - my)] == None and state[(col - 1, row + 1)] == op_piece):
                                ret[(1, 1, my)] += 1
                        # top-right of the counter-diagonal    
                        elif col + my > GRID_NUM and col - 1 >= 0 and row + 1 <= GRID_NUM:   
                            # (1, 1, my)
                            if state[(col - 1, row + 1)] == None:
                                ret[(1, 1, my)] += 1
                        # bottom-left of the counter-diagonal
                        elif col + my >= 0 and row - my >= 0 and row + 1 > GRID_NUM:
                            # (1, 1, my)
                            if state[(col + my, row - my)] == None:
                                ret[(1, 1, my)] += 1
                    elif my == 5:
                        # win state
                        ret[(1, 5)] += 1
                else: 
                    my = 0
                
                if (state[(col, row)] == op_piece): 
                    op += 1
                    if 2 <= op < 5:
                        # middle of the counter-diagonal
                        if col + op <= GRID_NUM and row - op >= 0 and col - 1 >= 0 and row + 1 <= GRID_NUM:
                            # (-1, 0, op)
                            if state[(col + op, row - op)] == None and state[(col - 1, row + 1)] == None:
                                ret[(-1, 0, op)] += 1
                            # (-1, 1, op)
                            elif (state[(col + op, row - op)] == op_piece and state[(col - 1, row + 1)] == None) or (state[(col + op, row - op)] == None and state[(col - 1, row + 1)] == my_piece):
                                ret[(-1, 1, op)] += 1
                        # top-right of the counter-diagonal    
                        elif col + op > GRID_NUM and col - 1 >= 0 and row + 1 <= GRID_NUM:   
                            # (-1, 1, op)
                            if state[(col - 1, row + 1)] == None:
                                ret[(-1, 1, op)] += 1
                        # bottom-left of the counter-diagonal
                        elif col + op >= 0 and row - op >= 0 and row + 1 > GRID_NUM:
                            # (-1, 1, op)
                            if state[(col + op, row - op)] == None:
                                ret[(-1, 1, op)] += 1
                    elif op == 5:
                        # lose state
                        ret[(-1, 5)] += 1
                else: 
                    op = 0
                if row + 1 < GRID_NUM:
                    row += 1
        # top start
        for col_ in range(GRID_NUM + 1):
            col = col_
            my = 0
            op = 0
            for row in range(col_ + 1):
                if (state[(col, row)] == my_piece): 
                    my += 1
                    if 2 <= my < 5:
                        # middle of the counter-diagonal
                        if col + my <= GRID_NUM and row - my >= 0 and col - 1 >= 0 and row + 1 <= GRID_NUM:
                            # (1, 0, my)
                            if state[(col + my, row - my)] == None and state[(col - 1, row + 1)] == None:
                                ret[(1, 0, my)] += 1
                            # (1, 1, my)
                            elif (state[(col + my, row - my)] == op_piece and state[(col - 1, row + 1)] == None) or (state[(col + my, row - my)] == None and state[(col + 1, row + 1)] == op_piece):
                                ret[(1, 1, my)] += 1
                        # top-right of the counter-diagonal
                        elif col + my <= GRID_NUM and row - my < 0 and col - 1 >= 0 and row + 1 <= GRID_NUM:   
                            # (1, 1, my)
                            if state[(col - 1, row + 1)] == None:
                                ret[(1, 1, my)] += 1
                        # bottom-left of the counter-diagonal
                        elif col + my >= 0 and row - my >= 0 and col - 1 < 0 and row + 1 <= GRID_NUM:
                            # (1, 1, my)
                            if state[(col + my, row - my)] == None:
                                ret[(1, 1, my)] += 1
                    elif my == 5:
                        # win state
                        ret[(1, 5)] += 1
                else: 
                    my = 0

                if (state[(col, row)] == op_piece): 
                    op += 1
                    if 2 <= op < 5:
                        # middle of the diagonal
                        if col + op >= 0 and row - op >= 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:
                            # (-1, 0, op)
                            if state[(col - op, row - op)] == None and state[(col + 1, row + 1)] == None:
                                ret[(-1, 0, op)] += 1
                            # (-1, 1, op)
                            elif (state[(col - op, row - op)] == my_piece and state[(col + 1, row + 1)] == None) or (state[(col - op, row - op)] == None and state[(col + 1, row + 1)] == my_piece):
                                ret[(-1, 1, op)] += 1
                        # top-right of the counter-diagonal    
                        elif col - op >= 0 and row - op < 0 and col + 1 <= GRID_NUM and row + 1 <= GRID_NUM:   
                            # (-1, 1, op)
                            if state[(col + 1, row + 1)] == None:
                                ret[(-1, 1, op)] += 1
                        # bottom-left of the counter-diagonal
                        elif col - op >= 0 and row - op >= 0 and col + 1 > GRID_NUM and row + 1 <= GRID_NUM:
                            # (-1, 1, op)
                            if state[(col - op, row - op)] == None:
                                ret[(-1, 1, op)] += 1
                    elif op_piece == 5:
                        # lose state
                        ret[(-1, 5)] += 1
                else: 
                    op = 0
                if col - 1 >= 0:
                    col -= 1

        state[action] = None
        return ret

    def getQValue(self, state, action):
        weights = self.getWeights()
        ret = 0.0
        features = self.getFeatures(state, action)
        for feature in features:
            ret += features[feature] * weights[feature]
        return ret
    
    def update(self, state, action, nextState, reward):
        features = self.getFeatures(state, action)
        if self.gameState.isWin(self.piece):
            diff = reward + self.discount * 100 - self.getQValue(state, action)
            for feature in self.getWeights():
                if feature[0] == 1:
                    self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]
        elif self.gameState.isLost(self.piece):
            diff = reward + self.discount * -100 - self.getQValue(state, action)
            for feature in self.getWeights():
                if feature[0] == -1:
                    self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]
        else:
            diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
            for feature in self.getWeights():
                self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]

        self.weight_normalization()
    
    def weight_normalization(self):
        weights = self.getWeights()
        min_ = 99999999
        max_ = -99999999
        sum_nega = 0
        sum_posi = 0
        for feature in weights:
            if weights[feature] < min_:
                min_ = weights[feature]
            elif weights[feature] > max_:
                max_ = weights[feature]
            
            if weights[feature] < 0:
                sum_nega -= weights[feature]
            elif weights[feature] > 0:
                sum_posi += weights[feature]
            
        if min_ < -100:
            for feature in weights:
                if weights[feature] < -100:
                    weights[feature] /= sum_nega
                    weights[feature] *= 100

        if max_ > 100:
            for feature in weights:
                if weights[feature] > 100:
                    weights[feature] /= sum_posi
                    weights[feature] *= 100


class DQN_Agent (BasicAgent):
    '''DQN Agent'''
    typeName = "DQNAgent"

    def __init__(self, piece, gameState, isGUI = True, name:str='DQN_Player', isTraining=True):
        super().__init__(piece, gameState, name=name)
        self.isGUI = isGUI
        self.boardSize = len(gameState.getGridsAsList())
        self.DQNClass = DQNClass(boardWidth=gameState.boardWidth, boardHeight=gameState.boardHeight, learningRate=0.1,
                                    memorySize=10000, targetUpdateRounds=100, batchSize=10, netType='CNN')
        self.trainingRounds = 300000

        self.isTraining = isTraining
        if not self.isTraining: self.DQNClass.evaluationNet = torch.load(DQN_MCTS_88_Trained_MODEL_PATH)

    def getSuccessors (self):
        return self.gameState.getAvailableGrids ()
        Helper.NOT_REACHED()

    def getObservation(self):
        return self.gameState.getGridAsNumpyList()
        Helper.NOT_REACHED()
    
    def getTrainingRounds(self):
        return self.trainingRounds
        Helper.NOT_REACHED()
        
    def getDecision(self):
        '''Random Agent randomly pick one available.'''
        successors = self.getSuccessors()
        observation = self.getObservation()
        return self.DQNClass.chooseAction(observation, successors)
        Helper.NOT_REACHED()
        
    def takeAction (self, action):
        '''putPiece'''
        Xpos, Ypos = action
        self.putPiece(self.gameState, Xpos, Ypos, self.isGUI)
        return None
        Helper.NOT_REACHED()

class MinimaxAgent (BasicAgent):
    '''Minimax Agent'''
    typeName = "MinimaxAgent"
    def __init__(self, piece, gameState, depth, isGUI = True, name:str= "MinimaxAgent"):
        super().__init__(piece, gameState, name=name) 
        self.depth = depth
        self.isGUI = isGUI

    def generate_state_string_of_one_position(self, row, column, state):
        black = Piece.BLACK
        white = Piece.WHITE

        All_direction = []
        #string four line of one position (row, column)
        #for row
        row_str = ''
        for j in range((GRID_NUM + 1)):
            curr = state[(GRID_NUM + 1)*row + j]
            if curr[1] == None:
                row_str = row_str + '0'
            elif curr[1] == black:
                row_str = row_str + 'b'
            elif curr[1] == white:
                row_str = row_str + 'w'
        All_direction.append(row_str)

        #for column
        col_str = ''
        for j in range((GRID_NUM + 1)):
            curr = state[(GRID_NUM + 1) * j + column]
            if curr[1] == None:
                col_str = col_str + '0'
            elif curr[1] == black:
                col_str = col_str + 'b'
            elif curr[1] == white:
                col_str = col_str + 'w'
        All_direction.append(col_str)
        

        #for left diagonal
        sum_rc = row + column
        temp1 = ''
        for j in range((GRID_NUM + 1)):
            k = sum_rc - j
            if(0 <= k <= GRID_NUM):
                curr = state[(GRID_NUM + 1) * j + k]
                if curr[1] == None:
                    temp1 = temp1 + '0'
                elif curr[1] == black:
                    temp1 = temp1 + 'b'
                elif curr[1] == white:
                    temp1 = temp1 + 'w'
        All_direction.append(temp1)


        #for right diagonal
        minus_rc = row - column
        temp2 = ''
        for j in range((GRID_NUM + 1)):
            k = j - minus_rc
            if(0 <= k <= GRID_NUM):
                curr = state[(GRID_NUM + 1) * j + k]
                if curr[1] == None:
                    temp2 = temp2 + '0'
                elif curr[1] == black:
                    temp2 = temp2 + 'b'
                elif curr[1] == white:
                    temp2 = temp2 + 'w'
        All_direction.append(temp2)

        return All_direction

    
    def value_of_one_position(self, all_direction_list, color):
        #Basic chess mode and corresponding value for black side
        black_type_array = []
        white_type_array = []

        #连五
        FIVE_value = 600000
        FIVE_type = "bbbbb"
        black_type_array.append(FIVE_type)

        #活四
        FOUR_value = 20000
        FOUR_type = "0bbbb0"
        black_type_array.append(FOUR_type)

        #眠四
        BLOCK_FOUR_value = 1000
        BLOCK_FOUR_type_1 = "wbbbb0"
        black_type_array.append(BLOCK_FOUR_type_1)
        BLOCK_FOUR_type_2 = "0bbbbw"
        black_type_array.append(BLOCK_FOUR_type_2)
        BLOCK_FOUR_type_3 = "b0bbb"
        black_type_array.append(BLOCK_FOUR_type_3)
        BLOCK_FOUR_type_4 = "bbb0b"
        black_type_array.append(BLOCK_FOUR_type_4)
        BLOCK_FOUR_type_5 = "bb0bb"
        black_type_array.append(BLOCK_FOUR_type_5)

        #活三
        THREE_value = 1000
        THREE_type_1 = "00bbb00"
        black_type_array.append(THREE_type_1)
        THREE_type_2 = "w0bbb00"
        black_type_array.append(THREE_type_2)
        THREE_type_3 = "00bbb0w"
        black_type_array.append(THREE_type_3)
        THREE_type_4 = "0bb0b0"
        black_type_array.append(THREE_type_4)
        THREE_type_5 = "0b0bb0"
        black_type_array.append(THREE_type_5)

        #眠三
        BLOCK_THREE_value = 100
        BLOCK_THREE_type_1 = "wbbb00"
        black_type_array.append(BLOCK_THREE_type_1)
        BLOCK_THREE_type_2 = "00bbbw"
        black_type_array.append(BLOCK_THREE_type_2)
        BLOCK_THREE_type_3 = "wbb0b0"
        black_type_array.append(BLOCK_THREE_type_3)
        BLOCK_THREE_type_4 = "0b0bbw"
        black_type_array.append(BLOCK_THREE_type_4)
        BLOCK_THREE_type_5 = "wb0bb0"
        black_type_array.append(BLOCK_THREE_type_5)
        BLOCK_THREE_type_6 = "0bb0bw"
        black_type_array.append(BLOCK_THREE_type_6)
        BLOCK_THREE_type_7 = "w0bbb0w"
        black_type_array.append(BLOCK_THREE_type_7)


        #活二
        TWO_value = 100
        TWO_type_1 = "00bb00"
        black_type_array.append(TWO_type_1)
        TWO_type_2 = "0b0b0"
        black_type_array.append(TWO_type_2)
        TWO_type_3 = "b00b"
        black_type_array.append(TWO_type_3)
        

        #眠二
        BLOCK_TWO_value =  10
        BLOCK_TWO_type_1 = "wbb000"
        black_type_array.append(BLOCK_TWO_type_1)
        BLOCK_TWO_type_2 = "000bbw"
        black_type_array.append(BLOCK_TWO_type_2)
        BLOCK_TWO_type_3 = "wb0b00"
        black_type_array.append(BLOCK_TWO_type_3)
        BLOCK_TWO_type_4 = "00b0bw"
        black_type_array.append(BLOCK_TWO_type_4)
        BLOCK_TWO_type_5 = "wb00b0"
        black_type_array.append(BLOCK_TWO_type_5)
        BLOCK_TWO_type_6 = "0b00bw"
        black_type_array.append(BLOCK_TWO_type_6)

        #活一
        ONE_value = 10
        ONE_type_1 = "0b0"
        black_type_array.append(ONE_type_1)

        #眠一
        BLOCK_ONE_value = 1
        BLOCK_ONE_type_1 = "wb0"
        black_type_array.append(BLOCK_ONE_type_1)
        BLOCK_ONE_type_2 = "0bw"
        black_type_array.append(BLOCK_ONE_type_2)


        #construct white_type_array
        for item in black_type_array:
            for i in range(len(item)):
                if item[i] == 'w':
                    item = item[:i] + 'b' + item[i+1:]
                elif item[i] == 'b':
                    item = item[:i] + 'w' + item[i+1:]
            white_type_array.append(item)
        
        #从五连到眠一匹配  匹配到的用特殊字符代替
        #extract every row, column and diagonal
        count = 0
        if(color == "black"):
            for i in range(4):
                for j in range(31):
                    num = len(re.findall(black_type_array[j], all_direction_list[i]))
                    if(j == 0):
                        count += num * FIVE_value
                    elif(j == 1):
                        count += num * FOUR_value
                    elif(j >= 2 and j <= 6):
                        count += num * BLOCK_FOUR_value
                        if(j >= 4):
                            count += num * FOUR_value/5
                    elif(j >= 7 and j <= 11):
                        count += num * THREE_value
                    elif(j >= 12 and j <= 18):
                        count += num * BLOCK_THREE_value
                    elif(j >= 19 and j <= 21):
                        count += num * TWO_value
                    elif(j >= 22 and j <= 27):
                        count += num * BLOCK_TWO_value
                    elif(j == 28):
                        count += num * ONE_value
                    else:
                        count += num * BLOCK_ONE_value
                    all_direction_list[i].replace(black_type_array[j], '-')
        
        elif(color == "white"):
            for i in range(4):
                for j in range(31):
                    num = len(re.findall(white_type_array[j], all_direction_list[i]))
                    if(j == 0):
                        count += num * FIVE_value
                    elif(j == 1):
                        count += num * FOUR_value
                    elif(j >= 2 and j <= 6):
                        count += num * BLOCK_FOUR_value
                        if(j >= 4):
                            count += num * FOUR_value/5
                    elif(j >= 7 and j <= 11):
                        count += num * THREE_value
                    elif(j >= 12 and j <= 18):
                        count += num * BLOCK_THREE_value
                    elif(j >= 19 and j <= 21):
                        count += num * TWO_value
                    elif(j >= 22 and j <= 27):
                        count += num * BLOCK_TWO_value
                    elif(j == 28):
                        count += num * ONE_value
                    else:
                        count += num * BLOCK_ONE_value
                    all_direction_list[i].replace(black_type_array[j], '-')

        return count


    def evaluationFunction(self, state): 
        #traversal all positions and sum the value together
        black = Piece.BLACK
        white = Piece.WHITE

        black_value = 0
        white_value = 0
        #traversal the grid
        for i in range(GRID_NUM + 1):
            for j in range(GRID_NUM + 1):
                state_string_list = self.generate_state_string_of_one_position(i, j, state)
                if state[i*(GRID_NUM + 1) + j][1] == black:
                    black_value += self.value_of_one_position(state_string_list, "black")
                elif state[i*(GRID_NUM + 1) + j][1] == white:
                    white_value += self.value_of_one_position(state_string_list, "white")
                else:
                    continue
        
        return (black_value - 2.5*white_value, white_value - 2.5*black_value) 


    def terminal(self, state):
        b5 = "bbbbb"
        w5 = "wwwww"
        black = Piece.BLACK
        white = Piece.WHITE

        row = []
        for i in range((GRID_NUM + 1)):
            row_str = ''
            for j in range((GRID_NUM + 1)):
                curr = state[(GRID_NUM + 1)*i + j]
                if curr[1] == None:
                    row_str = row_str + '0'
                elif curr[1] == black:
                    row_str = row_str + 'b'
                elif curr[1] == white:
                    row_str = row_str + 'w'
            row.append(row_str)
        
        count_b = 0
        count_w = 0
        for item in row:
            count_b =  len(re.findall(b5, item))
            count_w =  len(re.findall(w5, item))
        if count_w != 0 or count_b != 0:
            return True

        
        col = []
        for i in range((GRID_NUM + 1)):
            col_str = ''
            for j in range((GRID_NUM + 1)):
                curr = state[i + (GRID_NUM + 1) * j]
                if curr[1] == None:
                    col_str = col_str + '0'
                elif curr[1] == black:
                    col_str = col_str + 'b'
                elif curr[1] == white:
                    col_str = col_str + 'w'
            col.append(col_str)

        for item in col:
            count_b = len(re.findall(b5, item))
            count_w = len(re.findall(w5, item))
        if count_w != 0 or count_b != 0:
            return True

        

        right_top = []
        for i in range(15):
            temp = ''
            for j in range(i):
                k = i - j
                if(0 <= k <= GRID_NUM and 0 <= j <= GRID_NUM):
                    curr = state[(GRID_NUM + 1) * j + k]
                    if curr[1] == None:
                        temp = temp + '0'
                    elif curr[1] == black:
                        temp = temp + 'b'
                    elif curr[1] == white:
                        temp = temp + 'w'
            right_top.append(temp)

        for item in right_top:
            count_b = len(re.findall(b5, item))
            count_w = len(re.findall(w5, item))
        if count_w != 0 or count_b != 0:
            return True


        right_down = []
        for i in range(15):
            temp = ''
            for j in range((GRID_NUM + 1)):
                k = j - i + GRID_NUM
                if(0 <= k <= GRID_NUM and 0 <= j <= GRID_NUM):
                    curr = state[(GRID_NUM + 1) * j + k]
                    if curr[1] == None:
                        temp = temp + '0'
                    elif curr[1] == black:
                        temp = temp + 'b'
                    elif curr[1] == white:
                        temp = temp + 'w'
            right_down.append(temp)
        
        for item in right_down:
            count_b =  len(re.findall(b5, item))
            count_w =  len(re.findall(w5, item))
        if count_w != 0 or count_b != 0:
            return True

      
        return False
    

    def getAvailableposition(self, state):
        resGrids = list(filter(lambda x: x[1] is None , state))
        return resGrids
        

    def generateSuccessor(self, state, action, color):
        temp = []
        for i in range(len(state)):
            if state[i][0] == action[0]:
                temp.append((state[i][0], color))
            else:
                temp.append(state[i])
        return temp
    
                
    '''State evaluation is for black piece
       So for black piece, it is max agent, for white piece, it is min agent'''

    #decision for black piece
    def max_value(self, state, depth, alpha, beta):  
        if(depth == self.depth or self.terminal(state) or self.getAvailableposition(state) == []):
            return (self.evaluationFunction(state)[1],None)
        v = -float('inf')
        result_action = None

        for action in self.getAvailableposition(state):
            next_state = self.generateSuccessor(state, action, Piece.BLACK)
            next_layer_value = self.min_value(next_state, depth + 1, alpha, beta)[0]
           
            if (next_layer_value > v):
               v = next_layer_value
               result_action = [action[0]]
            elif(next_layer_value == v):
                result_action.append(action[0])
            '''if v > beta:  
                return (v, action[0])
            alpha = max(v, alpha)'''
        return (v,result_action)


    #decision for white piece
    def min_value(self, state, depth, alpha, beta):
        if(depth == self.depth or self.terminal(state) or self.getAvailableposition(state) == []):
            return (self.evaluationFunction(state)[0], None)

        v = -float('inf')
        result_action = None
        for action in self.getAvailableposition(state):
            next_state = self.generateSuccessor(state, action, Piece.WHITE)
            next_layer_value = self.max_value(next_state, depth + 1, alpha, beta)[0]

            if(next_layer_value > v):  
                v = next_layer_value
                result_action = [action[0]]
            elif(next_layer_value == v):
                result_action.append(action[0])
            '''if v < alpha:
                return (v, action[0])
            beta = min(beta, v)'''
        return (v, result_action)


    def takeAction(self):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = float('inf')

        if self.piece == Piece.BLACK:
            action = random.choice(self.max_value(self.gameState.getGridsAsList(), 0, alpha, beta)[1])
            self.putPiece(self.gameState, action[0], action[1], self.isGUI)
        
        elif self.piece == Piece.WHITE:
            action = random.choice(self.min_value(self.gameState.getGridsAsList(), 0, alpha, beta)[1])
            self.putPiece(self.gameState, action[0], action[1], self.isGUI)
        
class MCTSAgent(BasicAgent):
    '''MCTS Agent'''

    typeName = 'MCTSAgent'
    def __init__(self, piece, gameState, isGUI = True, name: str = 'MCTSPlayer', computeTime=1, maxActionsNum=1000):
        super().__init__(piece, gameState, name=name)
        self.isGUI = isGUI

        self.computeTime = computeTime
        self.maxActionsNum = maxActionsNum

        self.confident = 1.96
        self.equivalence = 10000 # calc beta
        self.max_depth = 1

    def getSuccessors (self):
        return self.gameState.getAvailableGrids ()
        Helper.NOT_REACHED()

    def getObservation(self):
        return self.gameState.getGridAsNumpyList()
        Helper.NOT_REACHED()
        
    def getDecision(self):
        '''Random Agent randomly pick one available.'''
        successors = self.getSuccessors()
        if len(successors) == 1:
            return successors[0][0]
        
        self.plays = {}      # key:(player, move), value:visited times
        self.wins = {}       # key:(player, move), value:win times
        self.plays_rave = {} # key:move, value:visited times
        self.wins_rave = {}  # key:move, value:{player: win times}

        # simulations = 0      # debug use
        begin = time.time()
        while time.time() - begin < self.computeTime:
            gameStateCopy = self.gameState.deepcopy()  # simulation will change board's states,
            self.runSimulation(gameStateCopy)
            # simulations += 1
        
        return self.selectOneAction(successors)
        Helper.NOT_REACHED()
    
    def runSimulation(self, gameState):
        plays = self.plays
        wins = self.wins
        plays_rave = self.plays_rave
        wins_rave = self.wins_rave

        player = gameState.getCurrentPiece()
        visited_states = set()
        expand = True
        winner = None
        for step in range(1, self.maxActionsNum + 1):
            successors = [succ[0] for succ in gameState.getAvailableGrids()]
            # print(successors)
            if all(plays.get((player, move)) for move in successors):
                value, move = max(
                    ((1-sqrt(self.equivalence/(3 * plays_rave[move] + self.equivalence))) * (wins[(player, move)] / plays[(player, move)]) +
                     sqrt(self.equivalence/(3 * plays_rave[move] + self.equivalence)) * (wins_rave[move][player] / plays_rave[move]) + 
                     sqrt(self.confident * log(plays_rave[move]) / plays[(player, move)]), move)
                    for move in successors)   # UCT RAVE  
            else:
                adjacents = []
                if len(successors) > gameState.boardWidth:
                    adjacents = self.adjacent_moves(gameState, plays)

                if len(adjacents):
                    move = choice(adjacents)
                else:
                    peripherals = []
                    for move in successors:
                        if not plays.get((player, move)):
                            peripherals.append(move)
                    move = choice(peripherals)

            self.putPieceSimulator(gameState, move[0], move[1], isGUI=False)
            # successors.remove(move)
            # Expand
            # add only one new child node each time
            if expand and (player, move) not in plays:
                expand = False
                plays[(player, move)] = 0
                wins[(player, move)] = 0
                if move not in plays_rave:
                    plays_rave[move] = 0
                if move in wins_rave:
                    wins_rave[move][player] = 0
                else:
                    wins_rave[move] = {player: 0}
                if step > self.max_depth:
                    self.max_depth = step

            visited_states.add((player, move))

            if gameState.isWin(player):
                winner = player
                break
            if gameState.isDraw():
                break
            
            player = gameState.getCurrentPiece()

        # Back-propagation
        for player, move in visited_states:
            if (player, move) in plays:
                plays[(player, move)] += 1 # all visited moves
                if player == winner:
                    wins[(player, move)] += 1 # only winner's moves
            if move in plays_rave:
                plays_rave[move] += 1 # no matter which player
                if winner in wins_rave[move]:
                    wins_rave[move][winner] += 1 # each move and every player

    def selectOneAction(self, successors):
        successors = [succ[0] for succ in successors]
        percent_wins, move = max(
            (self.wins.get((self.piece, move), 0) /
             self.plays.get((self.piece, move), 1),
             move)
            for move in successors)

        return move

    def adjacent_moves(self, gameState, plays):
        width = gameState.boardWidth
        height = gameState.boardHeight

        moved = [grid[0] for grid in gameState.getUnavailableGrids()]
        adjacents = set()

        for m in moved:
            h = m[1]
            w = m[0]
            if w < width - 1:
                adjacents.add((w + 1, h)) # right
            if w > 0:
                adjacents.add((w - 1, h)) # left
            if h < height - 1:
                adjacents.add((w, h + 1)) # upper
            if h > 0:
                adjacents.add((w, h - 1)) # lower
            if w < width - 1 and h < height - 1:
                adjacents.add((w + 1, h + 1)) # upper right
            if w > 0 and h < height - 1:
                adjacents.add((w - 1, h + 1)) # upper left
            if w < width - 1 and h > 0:
                adjacents.add((w + 1, h - 1)) # lower right
            if w > 0 and h > 0:
                adjacents.add((w - 1, h - 1)) # lower left

        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get((gameState.getCurrentPiece(), move)):
                adjacents.remove(move)
        return adjacents

    def takeAction (self, action):
        '''putPiece'''
        Xpos, Ypos = action
        self.putPiece(self.gameState, Xpos, Ypos, self.isGUI)
        return None
        Helper.NOT_REACHED()

    def putPieceSimulator (self, gameState:object, Xpos:int, Ypos:int, isGUI:bool = True)->int:
        grids = gameState.getGrids ()
        if (grids[(Xpos, Ypos)] is not None):
            Helper.WARNING (f"{(Xpos, Ypos)} already has a piece.")
            return False
        else:
            grids[(Xpos, Ypos)] = gameState.getCurrentPiece()
            isWin = gameState.check (gameState.getCurrentPiece(), Xpos, Ypos)
            # return reward
            if (isWin):
                gameState.winPiece = gameState.getCurrentPiece()
                # print (Colored.cyan(DEFAULT_WIN_REFIX + f"{self.name} with {self.piece.value} piece wins!"))
            gameState.count += 1
            gameState.currPiece = gameState.getNextPlayer()
            return True
        Helper.NOT_REACHED ()