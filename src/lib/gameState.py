import tkinter as tk 
import numpy as np
from enum import Enum, unique
import sys
sys.path.append ('..')
sys.path.append ('.')
from lib.const import *
from lib.utils import *
from lib.chessBoard import ChessBoard, Window

import copy

# 2020/12/5 Limzh -- Add
# maintain gamestate. GUI updates itself depending on the current gamestate
class GameState (object):
    def __init__(self, window=None, isGUI=False, gridNum=GRID_NUM):
        self.isGUI = isGUI
        self.gridNum = gridNum
        self.count:int = 0
        if (isGUI):
            self.window:Window = window 
            self.chessBoard:ChessBoard = ChessBoard (window.root)
        self.currPiece:Piece = Piece.BLACK
        # grids
        self.grids:numpy = self.initialGrids (gridNum)
        # isWin flag
        # 2020/12/11 Limzh -- Add
        self.winPiece: Piece = None
        # the size of grid
        # 2021/1/5 Lidr -- Add
        self.boardWidth = gridNum + 1
        self.boardHeight = gridNum + 1
        return
        Helper.NOT_REACHED ()
    def initialGrids (self, gridNum = GRID_NUM):
        resGrids = {}
        for i in range (gridNum+1):
            for j in range (gridNum+1):
                resGrids[(i,j)] = None
        return resGrids
        Helper.NOT_REACHED ()
    def getGrids (self):
        return self.grids
        Helper.NOT_REACHED ()
    def getCurrentPiece (self):
        return self.currPiece
        Helper.NOT_REACHED ()
    def getCount (self):
        return self.count
        Helper.NOT_REACHED ()
    def getGridAsNumpyList (self):
        '''An alternative method to search for info about piece on board.
           Note that, the modification on this result is not consistent with self.grids'''
        resGrids = np.zeros(shape=(self.gridNum+1, self.gridNum+1),dtype=object)
        tmpGirdsList = self.getGridsAsList ()
        for i, item in enumerate (tmpGirdsList):
            pos, val = item
            x, y = pos
            resGrids[x, y] = val
        return resGrids
        Helper.NOT_REACHED ()
    def getGridsAsList (self):
        resGrids = list(self.grids.items())
        return resGrids
        Helper.NOT_REACHED ()
    def isWin (self, piece):
        return piece == self.winPiece
        Helper.NOT_REACHED ()
    def isLost (self, piece):
        return not self.isWin (piece)
        Helper.NOT_REACHED()
    def isDraw (self):
        return (len(self.getAvailableGrids()) == 0) \
                and (not self.isWin(Piece.BLACK)) \
                and (not self.isWin(Piece.WHITE))
        Helper.NOT_REACHED ()
    def reset (self):
        self.grids = self.initialGrids ()
        self.currPiece = Piece.BLACK
        self.winPiece = None
        if (self.isGUI):
            for i in range (1, self.count+1):
                self.chessBoard.canvas.delete ("chess"+str(i))
        self.count = 0
        return True
        Helper.NOT_REACHED ()
    def check (self, piece, Xpos, Ypos):
        if (self.checkRow(piece, Ypos) or self.checkCol(piece, Xpos)):
            return True
        elif (self.checkDiagonal (piece, Xpos, Ypos)):
            return True
        else:
            return False
    def checkDiagonal (self, piece, Xpos, Ypos):
        grids = self.getGridAsNumpyList ()
        count = 0
        # from left_upon to right_bottom
        tmpPos = min (Xpos, Ypos)
        newXpos, newYpos = Xpos-tmpPos, Ypos-tmpPos
        while True:
            if (newXpos >= self.gridNum+1 or newYpos >= self.gridNum+1): break
            if (grids[(newXpos, newYpos)] == piece): count += 1
            else: count = 0
            if (count >= 5): return True
            newXpos, newYpos = newXpos + 1, newYpos + 1
        count = 0
        # from left_bottom to right_upon
        tmpPos = min (Xpos, self.gridNum-Ypos)
        newXpos, newYpos = Xpos-tmpPos, Ypos+tmpPos
        while True:
            if (newXpos >= self.gridNum+1 or newYpos <= -1): break
            if (grids[(newXpos, newYpos)] == piece): count += 1
            else: count = 0
            if (count >= 5): return True
            newXpos, newYpos = newXpos + 1, newYpos - 1
        return False
        Helper.NOT_REACHED ()
            
    def checkRow (self, piece, Ypos):
        grids = self.getGridAsNumpyList ()
        count = 0
        # Firstly, check the row
        for x in range (self.gridNum+1):
            # update
            if (grids[(x, Ypos)] == piece): count += 1
            else: count = 0
            if (count >= 5):
                return True
        return False
    def checkCol (self, piece, Xpos):
        grids = self.getGridAsNumpyList ()
        count = 0
        # Firstly, check the row
        for y in range (self.gridNum+1):
            # update
            if (grids[(Xpos, y)] == piece): count += 1
            else: count = 0
            if (count >= 5):
                return True
        return False

    def putMsg (self, msg:str):
        '''This function should be called iff self.isGUI is true'''
        import tkinter.messagebox as msgbox 
        if (self.isGUI == False): 
            Helper.PANIC ('`putMsg` is called with isGUI false.')
        return msgbox.askyesno("Message", msg)
    
    def getAvailableGrids (self) -> list:
        ''' return a numpy list of available positions to put piece. (successors)'''
        grids = self.getGridsAsList ()
        resGrids = list (filter( lambda x: x[1] is None , grids))
        return resGrids
    
    def getUnavailableGrids (self) -> list:
        ''' return a numpy list of unavailable positions to put piece. '''
        grids = self.getGridsAsList ()
        resGrids = list (filter( lambda x: x[1] is not None , grids))
        return resGrids

    def getNextPlayer(self):
        if self.currPiece == Piece.BLACK:
            return Piece.WHITE
        elif self.currPiece == Piece.WHITE:
            return Piece.BLACK
        
        Helper.NOT_REACHED()

    def deepcopy(self):
        result = GameState(window=None, isGUI=False)
        result.currPiece = copy.deepcopy(self.currPiece)
        result.grids = copy.deepcopy(self.grids)
        result.winPiece = copy.deepcopy(self.winPiece)
        result.count = copy.deepcopy(self.count)
        
        return result
        

# 2020/12/5 Limzh -- Add 
# Test part, debug purpose.
def main ():
    from lib.agents import BasicAgent, HumanAgent
    from random import choice
    window = Window ()
    gameState = GameState (window, False)
    grids = gameState.getGrids ()
    grids[(1,1)] = Piece.BLACK
    avalGrids = gameState.getAvailableGrids ()
    print (choice (avalGrids)[0])
    # print (gameState.getAvailableGrids())
if __name__ == '__main__':
    main ()