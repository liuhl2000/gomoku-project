import numpy as np 
import pandas as pd 
import scipy as sp 
import argparse

from lib.const import *
from lib.utils import *
from lib.agents import *
from lib.gameState import *
from lib.chessBoard import *
from lib.controller import *


def main ():
    # window = Window (title="Minimax Median")
    # gameState = GameState (window, True)
    
    # gameState = GameState()
    # player1 = MCTSAgent (Piece.BLACK, gameState, isGUI=False, computeTime=0.001)
    # player2 = RandomAgent(Piece.WHITE, gameState, isGUI=False)
    # controller = Controller (gameState, player1, player2, "MCTSvsRandom")

    # player1 = DQN_Agent(Piece.BLACK, gameState, isGUI=False, isTraining=True)
    # player2 = MCTSAgent(Piece.WHITE, gameState, isGUI=False, computeTime=0.001)
    # player1.trainingRounds = 10000
    # trainer = Trainer(gameState, player1, player2, "Train_DQN_with_MCTS")

    # player1 = HumanAgent (Piece.BLACK, gameState, name="Jane")
    # player2 = DQN_Agent (Piece.WHITE, gameState, isGUI=True, isTraining=False)
    # player2 = MCTSAgent (Piece.WHITE, gameState, isGUI=True, computeTime=1)
    # player2 = MinimaxAgent(Piece.WHITE, gameState, 1)
    # controller = Controller(gameState, player1, player2, "vsMinimaxAI")
    # player2 = HumanAgent (Piece.WHITE, gameState, name="Reagan")
    # controller = Controller (gameState, player1, player2, "vsDQNAI")
    # controller = Controller (gameState, player1, player2, "vsDQNAI")
    # window.mainLoop ()    
    # # # ======= TRAIN ======
    # gameState = GameState(gridNum=7)
    # player1 = DQN_Agent(Piece.BLACK, gameState, isGUI=False, isTraining=True)
    # player2 = MCTSAgent(Piece.WHITE, gameState, isGUI=False, computeTime=0.001)
    # trainer = Trainer(gameState, player1, player2, "Train_DQN_with_MCTS")
    # gameState = GameState(window, True)
    # player1 = HumanAgent(Piece.BLACK, gameState, name="Reagan")
    # player2 = HumanAgent(Piece.WHITE, gameState, name="Jane")
    # Controller(gameState, player1, player2, "vsHuman")
    # window.mainLoop()
     # create window
    # window = Window()
    for i in [3]:
        gameState = GameState(isGUI=False)
        player1 = MinimaxAgent(Piece.WHITE, gameState, 1, isGUI=False)
        player2 = MCTSAgent(Piece.BLACK, gameState, isGUI=False, computeTime=i)
    # # player1 = DQN_Agent(Piece.WHITE, gameState, isGUI=False, isTraining=False)
    # # player2 = RandomAgent(Piece.BLACK, gameState, isGUI=False)
        # player2 = MinimaxAgent(Piece.BLACK, gameState, 1, isGUI=False)
        controller = Controller(gameState, player1, player2, "MCTSploting")
    #controller = Controller(gameState, player1, player2, "vsRandomAI")
    window.mainLoop ()    



if __name__ == "__main__":
    main ()