import tkinter as tk
import sys
sys.path.append ('..')
from lib.const import *
from lib.utils import *

class Window (object):
    def __init__(self, title = "Gomoku", width = WINDOW_WIDTH, height = WINDOW_HEIGHT):
        super(Window, self).__init__()
        self._setWindowSetting (width, height)
        self._createWindow (title)
    def _createWindow (self, title):
        self.root = tk.Tk ()
        self.root.title (title)
        self.root.geometry (str(self.width) + 'x' + str(self.height))
    def _setWindowSetting (self, width, height):
        self.width = width
        self.height = height
    def mainLoop (self):
        self.root.mainloop ()

# 2020/12/5 Limzh -- Add
class ChessBoard (object):
    '''
    Basic GUI setting for chessboard with drawing method.
    '''
    def __init__(self, window, width=750, height=750, startPoint = DRAW_LEFT_UPON, endPoint = DRAW_RIGHT_DOWN):
        print (Colored.blue (DEFAULT_GUI_PREFIX + " Building ChessBoard ..."))
        self.window = window
        assert (self._SetBasicSetting(width, height, startPoint, endPoint))
        assert (self._createCanvas (window))
        assert (self._draw_chessBoard ())
    
    def drawChess (self, piece, Xpos, Ypos, tag):
        assert (0 <= Xpos <= GRID_NUM)
        assert (0 <= Ypos <= GRID_NUM)
        assert (piece == Piece.BLACK or piece == Piece.WHITE)
        startX, startY = DRAW_LEFT_UPON
        x, y = startX + Xpos*self.gridSize, startY + Ypos*self.gridSize
        if (piece == Piece.BLACK):
            color = "black"
        elif (piece == Piece.WHITE):
            color = "white"
        else:
            Helper.PANIC ("Invalid piece type.")
        # draw a piece
        self.canvas.create_oval (x-16, y-16, x+16, y+16, fill = color, tags = tag)
        return True

    def _createCanvas (self, window, bg = "saddlebrown"):
        self.canvas = tk.Canvas (window, bg=bg, width=self.width, height=self.height)
        return True
        Helper.NOT_REACHED ()
    
    # 2020/12/5 Limzh -- Add
    # Accomplish `Inquriy` part of ChessBoard_GUI
    def _SetBasicSetting (self, width, height, startPoint, endPoint):
        '''
        Nothing fancy, set width, height, statrPoint, endPoint.
        '''
        # size setting
        if (width >= WINDOW_WIDTH): width = WINDOW_WIDTH
        if (height >= WINDOW_HEIGHT): height = WINDOW_HEIGHT
        self.width, self.height = width, height
        # grid setting
        self.startPoint, self.endPoint = startPoint, endPoint
        # sanity-check
        if (startPoint[0] != startPoint[1] and endPoint[0] != endPoint[1]):
            Helper.PANIC(" Invalid startPoint or endPoint. ")
        self.gridSize = (endPoint[0] - startPoint[0]) / GRID_NUM
        # `Star Point` plus `TianYuan`
        # `Star Point` should be (4, 4), (4, 12), (12, 4), (12, 12); `TianYuan` should be (8, 8)
        self.starPoints = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        return True
        Helper.NOT_REACHED ()
       
    def getWidth (self):
        return self.width
        Helper.NOT_REACHED ()
    def getHeight (self):
        return self.height
        Helper.NOT_REACHED ()
    def getStartPoint (self):
        return self.startPoint
        Helper.NOT_REACHED ()
    def getEndPoint (self):
        return self.endPoint
        Helper.NOT_REACHED ()
    def getGridSize (self):
        return self.gridSize
        Helper.NOT_REACHED ()
    # 2020/12/5 Limzh -- Add
    # Accomplished drawing part for ChessBoard_GUI
    def _draw_chessBoard (self):
        self.canvas.grid (row=0, column=9, rowspan=6)
        startX, startY = self.startPoint
        endX,   endY   = self.endPoint
        # draw lines
        for i in range (GRID_NUM+1):
            self.canvas.create_line (startX, (self.gridSize*i + startY), endX, (self.gridSize*i + startY))
            self.canvas.create_line (self.gridSize*i + startX, startY, self.gridSize*i + startX, endY)
        # draw points
        for i, point in enumerate(self.starPoints):
            x, y = point
            self.canvas.create_oval (self.gridSize*x + startX - 3 , self.gridSize*y + startY - 3,
                                self.gridSize*x + startX + 3, self.gridSize*y + startY + 3, fill= "black")
        # draw y-axis coordinate
        for i in range (GRID_NUM + 1):
            label = tk.Label (self.canvas, text = str(i+1), fg='black', bg='saddlebrown',width=2, anchor=tk.E)
            label.place (x = 60, y = 40*i+95)
        # draw x-axis coordinate
        x_axis = [chr(i) for i in range(65, 80)]
        for i, labelName in enumerate(x_axis):
            label = tk.Label (self.canvas, text = labelName, fg='black', bg='saddlebrown')
            label.place (x=self.gridSize*i + 95, y = 60)
        return True
        Helper.NOT_REACHED ()
            

# 2020/12/5 Limzh -- Add 
# Test part. the following part runs only in the local test setting for debug purpose.       
def main ():
    window = Window ()
    chessBoard = ChessBoard (window.root)
    window.mainLoop ()

if __name__ == "__main__":
    main ()
