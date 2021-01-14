import numpy as np 
import pandas as pd 
import scipy as sp 
import sys
import pickle
sys.path.append ('.')
sys.path.append ('..')

from lib.const import *
from lib.utils import *
from lib.agents import *
from lib.gameState import *
from lib.chessBoard import *

import matplotlib.pyplot as plt


class Controller (object):
    def __init__(self, gameState, player1, player2, name):
        self.gameState = gameState
        self.player1 = player1
        self.player2 = player2
        self.name = name
        if (name == "vsHuman"):
            gameState.chessBoard.canvas.bind ("<Button-1>", self.vsHuman)
        elif (name == "vsRandomAI"):
            self.vsRandomAI()
        elif (name == "vsQAgent"):
            self.vsQAgent()
        elif (name == "vsApproxQAgent"):
            self.vsApproxQAgent()
        elif (name == "vsDQNAI"):
            self.vsDQNAI()
        elif (name == "DQNvsRandom"):
            self.DQNvsRandom()
        elif (name == "vsMCTSAI"):
            self.vsMCTSAI()
        elif (name == "MCTSvsRandom"):
            self.MCTSvsRandom()
        elif (name == "DQNvsMCTS"):
            self.DQNvsMCTS()
        elif (name == "vsMinimaxAI"):
            self.vsMinimaxAI()
        elif (name == "MinimaxvsOther"):
            self.MinimaxvsOther()
        elif (name == "MCTSploting"):
            self.MCTSPloting()

    # 2020/12/11 Limzh -- Add
    def checkWin (self, isGUI = True):
        '''After each turn for action, we check it the gameState reaches the goal state.'''
        isReset = False
        if (self.gameState.isWin (self.player1.piece) or self.gameState.isWin (self.player2.piece)):
            if (isGUI): isReset = self.gameState.putMsg ("Reset?")
            else: 
                Helper.MESSAGE ("Reset? (y or n)")
                isReset = (input () == 'y')
            if (isReset): self.gameState.reset ()
            else: Helper.EXIT ("Exiting...")
        return None
        Helper.NOT_REACHED ()

    def vsHuman (self, event) -> None:
        assert (self.player1.typeName == "HUMAN" and self.player2.typeName == "HUMAN") 
   
        if self.gameState.currPiece == self.player1.piece:
            self.player1.getClickPos (event)
        else:
            self.player2.getClickPos (event)

        self.checkWin (isGUI = self.gameState.isGUI)
        return None
        Helper.NOT_REACHED ()
    # TODO: PlEASE ADD YOUR `CONTROLLER` HERE.

    def vsRandomAI (self):

        if (self.player1.typeName == "RandomAgent"):
            assert (self.player2.typeName == "HUMAN")
            humanAgent = self.player2
            randomAgent = self.player1
        elif (self.player2.typeName == "RandomAgent"):
            assert (self.player1.typeName == "HUMAN")
            humanAgent = self.player1
            randomAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `vsRandomAI` !")
        
        def getClickPos (event):
            isSuccess = humanAgent.getClickPos (event)
            if(isSuccess):
                randomAgent.takeAction ()
            self.checkWin ()
            if (self.gameState.currPiece == randomAgent.piece): randomAgent.takeAction ()

        self.gameState.chessBoard.canvas.bind ("<Button-1>", getClickPos)

        if (self.gameState.currPiece == randomAgent.piece): randomAgent.takeAction ()
        
        return None
        Helper.NOT_REACHED()
    
    def vsQAgent(self):
        if (self.player1.typeName == "QAgent"):
            assert (self.player2.typeName == "HUMAN")
            humanAgent = self.player2
            QAgent = self.player1
        elif (self.player2.typeName == "QAgent"):
            assert (self.player1.typeName == "HUMAN")
            humanAgent = self.player1
            QAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `vsQAgent` !")

        def getClickPos (event):
            isSuccess = humanAgent.getClickPos (event)
            if(isSuccess):
                action = QAgent.getDecision()
                QAgent.takeAction(action)
            self.checkWin ()
            if (self.gameState.currPiece == QAgent.piece):
                action = QAgent.getDecision()
                QAgent.takeAction(action)

        self.gameState.chessBoard.canvas.bind ("<Button-1>", getClickPos)

        if (self.gameState.currPiece == QAgent.piece): 
            action = QAgent.getDecision()
            QAgent.takeAction(action)
        
        return None
        Helper.NOT_REACHED ()
    
    def vsApproxQAgent(self):
        if (self.player1.typeName == "ApproximateQAgent"):
            assert (self.player2.typeName == "HUMAN")
            humanAgent = self.player2
            ApproxQAgent = self.player1
        elif (self.player2.typeName == "ApproximateQAgent"):
            assert (self.player1.typeName == "HUMAN")
            humanAgent = self.player1
            ApproxQAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `vsApproxQAgent` !")

        def getClickPos (event):
            isSuccess = humanAgent.getClickPos (event)
            if(isSuccess):
                action = ApproxQAgent.getDecision()
                ApproxQAgent.takeAction(action)
            self.checkWin ()
            if (self.gameState.currPiece == ApproxQAgent.piece):
                action = ApproxQAgent.getDecision()
                ApproxQAgent.takeAction(action)

        self.gameState.chessBoard.canvas.bind ("<Button-1>", getClickPos)

        if (self.gameState.currPiece == ApproxQAgent.piece): 
            action = ApproxQAgent.getDecision()
            ApproxQAgent.takeAction(action)
        
        return None
        Helper.NOT_REACHED ()
    
    def vsDQNAI (self):

        if (self.player1.typeName == "DQNAgent"):
            assert (self.player2.typeName == "HUMAN")
            humanAgent = self.player2
            DQNAgent = self.player1
        elif (self.player2.typeName == "DQNAgent"):
            assert (self.player1.typeName == "HUMAN")
            humanAgent = self.player1
            DQNAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `vsDQNAI` !")
        
        def getClickPos (event):
            isSuccess = humanAgent.getClickPos (event)
            if(isSuccess):
                DQNAgent.takeAction (DQNAgent.getDecision())
            self.checkWin ()
            if (self.gameState.currPiece == DQNAgent.piece): DQNAgent.takeAction (DQNAgent.getDecision())

        self.gameState.chessBoard.canvas.bind ("<Button-1>", getClickPos)

        if (self.gameState.currPiece == DQNAgent.piece): DQNAgent.takeAction (DQNAgent.getDecision())
        
        return None
        Helper.NOT_REACHED()
    
    def DQNvsMCTS (self):
        from tqdm import tqdm
        if (self.player1.typeName == "DQNAgent"):
            assert (self.player2.typeName == "MCTSAgent")
            MCTSAgent = self.player2
            DQNAgent = self.player1
        elif (self.player2.typeName == "DQNAgent"):
            assert (self.player1.typeName == "MCTSAgent")
            MCTSAgent = self.player1
            DQNAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `DQNvsMCTS` !")
        
        DQNWinningNum = 0
        for episode in tqdm(range(100)):
            self.gameState.reset()
            while not self.gameState.isWin(DQNAgent.piece) and not self.gameState.isWin(MCTSAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == MCTSAgent.piece:
                    MCTSAgent.takeAction(MCTSAgent.getDecision())
                else:
                    DQNAgent.takeAction(DQNAgent.getDecision())
            if self.gameState.isWin(DQNAgent.piece): DQNWinningNum += 1

        print(DQNWinningNum / 100)
        return None
        Helper.NOT_REACHED()
        
    def DQNvsRandom (self):
        from tqdm import tqdm
        if (self.player1.typeName == "DQNAgent"):
            assert (self.player2.typeName == "RandomAgent")
            RandomAgent = self.player2
            DQNAgent = self.player1
        elif (self.player2.typeName == "DQNAgent"):
            assert (self.player1.typeName == "RandomAgent")
            RandomAgent = self.player1
            DQNAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `DQNvsRandom` !")
        
        DQNWinningNum = 0
        for episode in tqdm(range(100)):
            self.gameState.reset()
            while not self.gameState.isWin(DQNAgent.piece) and not self.gameState.isWin(RandomAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == RandomAgent.piece:
                    RandomAgent.takeAction()
                else:
                    DQNAgent.takeAction(DQNAgent.getDecision())
            if self.gameState.isWin(DQNAgent.piece): DQNWinningNum += 1

        print(DQNWinningNum / 100)
        return None
        Helper.NOT_REACHED()
        
    def vsMCTSAI(self):
        if (self.player1.typeName == "MCTSAgent"):
            assert (self.player2.typeName == "HUMAN")
            humanAgent = self.player2
            MCTSAgent = self.player1
        elif (self.player2.typeName == "MCTSAgent"):
            assert (self.player1.typeName == "HUMAN")
            humanAgent = self.player1
            MCTSAgent = self.player2
        else:
            Helper.PANIC("Invalid players for `vsMCTSAI` !")
            
        def getClickPos (event):
            isSuccess = humanAgent.getClickPos (event)
            if(isSuccess):
                MCTSAgent.takeAction (MCTSAgent.getDecision())
            self.checkWin ()
            if (self.gameState.currPiece == MCTSAgent.piece): MCTSAgent.takeAction (MCTSAgent.getDecision())

        self.gameState.chessBoard.canvas.bind ("<Button-1>", getClickPos)

        if (self.gameState.currPiece == MCTSAgent.piece): MCTSAgent.takeAction (MCTSAgent.getDecision())
        
        return None
        Helper.NOT_REACHED()

    def MCTSvsRandom (self):
        from tqdm import tqdm
        if (self.player1.typeName == "MCTSAgent"):
            assert (self.player2.typeName == "RandomAgent")
            RandomAgent = self.player2
            MCTSAgent = self.player1
        elif (self.player2.typeName == "MCTSAgent"):
            assert (self.player1.typeName == "RandomAgent")
            RandomAgent = self.player1
            MCTSAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `MCTSvsRandom` !")
        
        MCTSWinningNum = 0
        for episode in tqdm(range(100)):
            self.gameState.reset()
            while not self.gameState.isWin(MCTSAgent.piece) and not self.gameState.isWin(RandomAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == RandomAgent.piece:
                    RandomAgent.takeAction()
                else:
                    MCTSAgent.takeAction(MCTSAgent.getDecision())
            if self.gameState.isWin(MCTSAgent.piece): MCTSWinningNum += 1

        print(MCTSWinningNum / 100)
        return None
        Helper.NOT_REACHED()


    def vsMinimaxAI(self):
        if (self.player1.typeName == "MinimaxAgent"):
            assert (self.player2.typeName == "HUMAN")
            humanAgent = self.player2
            MinimaxAgent = self.player1
        elif (self.player2.typeName == "MinimaxAgent"):
            assert (self.player1.typeName == "HUMAN")
            humanAgent = self.player1
            MinimaxAgent = self.player2
        else:
            Helper.PANIC ("Invalid players for `vsMinimax` !")
        
        def getClickPos (event):
            isSuccess = humanAgent.getClickPos (event)
            if(isSuccess):
                MinimaxAgent.takeAction()
            self.checkWin ()
            if (self.gameState.currPiece == MinimaxAgent.piece): MinimaxAgent.takeAction()

        self.gameState.chessBoard.canvas.bind ("<Button-1>", getClickPos)

        if(self.gameState.currPiece == MinimaxAgent.piece): MinimaxAgent.takeAction()
        
        return None
        Helper.NOT_REACHED ()


    def MinimaxvsOther(self):
        from tqdm import tqdm
        MinimaxAgent = self.player1
        otherAgent = self.player2
        
        MinimaxWinningNum = 0
        otherWinningNum = 0
        for episode in tqdm(range(100)):
            self.gameState.reset()
            while not self.gameState.isWin(MinimaxAgent.piece) and not self.gameState.isWin(otherAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == otherAgent.piece:
                    otherAgent.takeAction(otherAgent.getDecision())
                else:
                    MinimaxAgent.takeAction(MinimaxAgent.getDecision())
            if self.gameState.isWin(MinimaxAgent.piece): MinimaxWinningNum += 1
            if self.gameState.isWin(otherAgent.piece): otherWinningNum += 1

            print('------------------MCTS(0.1)vsMCTS(5) 15*15--------------------------------')
            print("MCTS(0.1) Winning Rate:", MinimaxWinningNum, "MCTS(5) Winning Rate:", otherWinningNum)
            # print("Random Winning Rate:", otherWinningNum)
            print("--------------------------------------------------------------------------")
        return None
        Helper.NOT_REACHED()
    
    def MCTSPloting(self):
        from tqdm import tqdm
        MinimaxAgent = self.player1
        MCTSAgent = self.player2
        
        MinimaxWinningNum = 0
        MCTSWinningNum = 0

        for episode in tqdm(range(50)):
            self.gameState.reset()
            while not self.gameState.isWin(MinimaxAgent.piece) and not self.gameState.isWin(MCTSAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == MCTSAgent.piece:
                    MCTSAgent.takeAction(MCTSAgent.getDecision())
                else:
                    MinimaxAgent.takeAction()
            if self.gameState.isWin(MinimaxAgent.piece): MinimaxWinningNum += 1
            if self.gameState.isWin(MCTSAgent.piece): MCTSWinningNum += 1

        print('------------------Minimax vs MCTS(3) 8*8--------------------------------')
        print("Minimax Winning Rate:", MinimaxWinningNum/50, "MCTS Winning Rate:", MCTSWinningNum/50)
        print("--------------------------------------------------------------------------")
        return None
        Helper.NOT_REACHED()


class Trainer (object):
    def __init__(self, gameState, player1, player2, name, sampleNum = 100):
        self.gameState = gameState
        self.player1 = player1 # trainee
        self.player2 = player2 # trainer
        self.name = name
        self.sampleNum = sampleNum
        if (name == "Train_Q_with_Random"):
            self.training_result = self.Q_Trainer_Random()
            if self.training_result:
                self.player1.epsilon = 0.0
                with open(r"QValues.txt", "wb") as f:
                    pickle.dump(self.player1.QValues, f)
        elif (name == "Train_Approx_Q_with_Random"):
            self.training_result = self.Approx_Q_Trainer_Random()
            if self.training_result:
                self.player1.epsilon = 0.0
                with open(r"R_weights.txt", "wb") as f:
                    pickle.dump(self.player1.weights, f)
        elif (name == "Train_Approx_Q_with_Self"):
            self.training_result = self.Approx_Q_Trainer_Self()
            if self.training_result:
                self.player1.epsilon = 0.0
                with open(r"S_weights.txt", "wb") as f:
                    pickle.dump(self.player1.weights, f)
        elif (name == 'Train_DQN_with_Random'):
            self.training_result = self.DQN_Trainer_Random()
        elif (name == 'Train_DQN_with_Last_DQN'):
            self.training_result = self.DQN_Trainer_Last_DQN()
        elif (name == 'Train_DQN_with_MCTS'):
            self.training_result = self.DQN_Trainer_MCTS()

    def getReward(self, player):
        i = 0
        win = 0
        window = Window ()
        while i < self.sampleNum:
            newGame = GameState (window, False)
            newGrids = copy.deepcopy(self.gameState.getGrids())
            newGame.grids = newGrids          
            Q_Random_Agent = RandomAgent(player.piece, newGame, player.name, False, True)
            Random_Agent = RandomAgent(self.player2.piece, newGame, self.player2.name, False, True)
            while not newGame.isWin(self.player1.piece) and not newGame.isWin(self.player2.piece) and not newGame.isDraw():
                if newGame.currPiece == Q_Random_Agent.piece:
                    Q_Random_Agent.takeAction()
                else:
                    Random_Agent.takeAction()
            if newGame.isWin(Q_Random_Agent.piece):
                win += 1
            i += 1
        winningRate = win/self.sampleNum
        return winningRate
        Helper.NOT_REACHED()

    def Q_Trainer_Random(self):
        QAgent = self.player1
        Random_Agent = self.player2
        numTraining = QAgent.getNumTraining()

        for episode in range(numTraining):
            self.gameState.reset()
            while not self.gameState.isWin(QAgent.piece) and not self.gameState.isWin(Random_Agent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == Random_Agent.piece:
                    Random_Agent.takeAction()
                else:
                    oldState = self.gameState.getState()
                    action, _ = QAgent.getDecision()
                    QAgent.takeAction(action)
                    newState = self.gameState.getState()

                    reward = self.getReward(QAgent)
                    QAgent.update(oldState, action, newState, reward)
        Helper.MESSAGE("Training Finished...")

        return True

    def Approx_Q_Trainer_Random(self):
        Approx_QAgent = self.player1
        Random_Agent = self.player2
        numTraining = Approx_QAgent.getNumTraining()

        for episode in range(numTraining):
            self.gameState.reset()
            while not self.gameState.isWin(Approx_QAgent.piece) and not self.gameState.isWin(Random_Agent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == Random_Agent.piece:
                    Random_Agent.takeAction()
                else:
                    oldState = self.gameState.getGridAsNumpyList()
                    action, _ = Approx_QAgent.getDecision()
                    Approx_QAgent.takeAction(action)
                    # If Approx_QAgent wins the game, stop after one iteration
                    if self.gameState.isWin(Approx_QAgent.piece):
                        reward = 100
                        # this is a winning state, should be checked in update()
                        newState = self.gameState.getGridAsNumpyList()
                        Approx_QAgent.update(oldState, action, newState, reward)
                        break
                    else:
                        Random_Agent.takeAction()
                        # If RandomAgent wins the game, stop after one iteration
                        if self.gameState.isWin(Random_Agent.piece):
                            reward = -100
                            # this is a losing state, should be checked in update()
                            newState = self.gameState.getGridAsNumpyList()
                            Approx_QAgent.update(oldState, action, newState, reward)
                            break
                        else:
                            reward = 0
                            # this is a normal state
                            newState = self.gameState.getGridAsNumpyList()
                            Approx_QAgent.update(oldState, action, newState, reward)
        Helper.MESSAGE("Training Finished...")
        return True
                        
    def Approx_Q_Trainer_Self(self):
        Approx_QAgent = self.player1
        numTraining = Approx_QAgent.getNumTraining()

        for episode in range(numTraining):
            self.gameState.reset()
            while not self.gameState.isWin(Approx_QAgent.piece) and not self.gameState.isLost(Approx_QAgent.piece) and not self.gameState.isDraw():
                oldState = self.gameState.getGridAsNumpyList()
                action = Approx_QAgent.getDecision()
                Approx_QAgent.takeAction(action)
                # If Approx_QAgent wins the game, stop after one iteration
                if self.gameState.isWin(Approx_QAgent.piece):
                    reward = 100
                    # this is a winning state, should be checked in update()
                    newState = self.gameState.getGridAsNumpyList()
                    Approx_QAgent.update(oldState, action, newState, reward)
                else:
                    # change to another side
                    Approx_QAgent.changeSide()
                    op_action = Approx_QAgent.getDecision()
                    Approx_QAgent.takeAction(op_action)
                    # back to black side
                    Approx_QAgent.changeSide()

                    if self.gameState.isLost(Approx_QAgent.piece):
                        reward = -100
                        # this is a losing state, should be checked in update()
                        newState = self.gameState.getGridAsNumpyList()
                        Approx_QAgent.update(oldState, action, newState, reward)
                    else:
                        reward = 0
                        # this is a normal state
                        newState = self.gameState.getGridAsNumpyList()
                        Approx_QAgent.update(oldState, action, newState, reward)
            if (episode+1) % 10 == 0:
                print("Weights after %d episodes:" % (episode+1), end="")
                print(self.player1.weights)
        Helper.MESSAGE("Training Finished...")
        return True

    def DQN_Trainer_Random(self):
        from tqdm import tqdm
        DQNAgent = self.player1
        RandomAgent = self.player2
        trainingRounds = DQNAgent.getTrainingRounds()
        
        step = 0
        DQNWinningNum = 0
        for episode in tqdm(range(trainingRounds)):
            self.gameState.reset()
            while not self.gameState.isWin(DQNAgent.piece) and not self.gameState.isWin(RandomAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == RandomAgent.piece:
                    RandomAgent.takeAction()
                else:
                    currentState = self.gameState.getGridAsNumpyList()
                    action = DQNAgent.getDecision()
                    actionIndex = action[0] * DQNAgent.DQNClass.boardHeight + action[1]
                    DQNAgent.takeAction(action)
                    nextState = self.gameState.getGridAsNumpyList()
                    if (self.gameState.isWin(DQNAgent.piece)):
                        reward = 1000
                    elif (self.gameState.isWin(RandomAgent.piece)):
                        reward = -1000
                    else:
                        reward = 1

                    DQNAgent.DQNClass.storeTransition(currentState, actionIndex, reward, nextState)

                    if step > DQNAgent.DQNClass.targetUpdateRounds and step % DQNAgent.DQNClass.batchSize == 0:
                        DQNAgent.DQNClass.learn()
                step += 1
            
            if self.gameState.isWin(DQNAgent.piece): DQNWinningNum += 1
            if episode % 100 == 99:
                print("Episode:", episode + 1, "Winning rate:", DQNWinningNum / 100)
                DQNWinningNum= 0

        Helper.MESSAGE("Training Finished...")
        torch.save(DQNAgent.DQNClass.evaluationNet, DQN_MODEL_PATH)
        return True

    def DQN_Trainer_Last_DQN(self):
        from tqdm import tqdm
        DQNAgent = self.player1
        lastDQNAgent = self.player2
        trainingRounds = DQNAgent.getTrainingRounds()

        step = 0
        DQNWinningNum = 0
        for episode in tqdm(range(trainingRounds)):
            self.gameState.reset()
            while not self.gameState.isWin(DQNAgent.piece) and not self.gameState.isWin(lastDQNAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == lastDQNAgent.piece:
                    lastDQNAgent.takeAction(lastDQNAgent.getDecision())
                else:
                    currentState = self.gameState.getGridAsNumpyList()
                    action = DQNAgent.getDecision()
                    actionIndex = action[0] * DQNAgent.DQNClass.boardHeight + action[1]
                    DQNAgent.takeAction(action)
                    nextState = self.gameState.getGridAsNumpyList()
                    if (self.gameState.isWin(DQNAgent.piece)):
                        reward = 1
                    elif (self.gameState.isWin(lastDQNAgent.piece)):
                        reward = -1
                    else:
                        reward = -0.01

                    DQNAgent.DQNClass.storeTransition(currentState, actionIndex, reward, nextState)

                    if step > DQNAgent.DQNClass.targetUpdateRounds and step % DQNAgent.DQNClass.batchSize == 0:
                        DQNAgent.DQNClass.learn()
                step += 1
            
            if self.gameState.isWin(DQNAgent.piece): DQNWinningNum += 1
            if episode % 100 == 99:
                print("Episode:", episode + 1, "Winning rate:", DQNWinningNum / 100)
                DQNWinningNum= 0

        Helper.MESSAGE("Training Finished...")
        torch.save(DQNAgent.DQNClass.evaluationNet, DQN_MODEL_PATH)
        return True

    def DQN_Trainer_MCTS(self):
        from tqdm import tqdm
        DQNAgent = self.player1
        MCTSAgent = self.player2
        trainingRounds = DQNAgent.getTrainingRounds()

        step = 0
        DQNWinningNum = 0
        DQNWinningGood = 0
        winnings = []
        for episode in tqdm(range(trainingRounds)):
            self.gameState.reset()
            while not self.gameState.isWin(DQNAgent.piece) and not self.gameState.isWin(MCTSAgent.piece) and not self.gameState.isDraw():
                if self.gameState.currPiece == MCTSAgent.piece:
                    MCTSAgent.takeAction(MCTSAgent.getDecision())
                else:
                    currentState = self.gameState.getGridAsNumpyList()
                    action = DQNAgent.getDecision()
                    actionIndex = action[0] * DQNAgent.DQNClass.boardHeight + action[1]
                    DQNAgent.takeAction(action)
                    nextState = self.gameState.getGridAsNumpyList()
                    if (self.gameState.isWin(DQNAgent.piece)):
                        reward = 1000
                    elif (self.gameState.isWin(MCTSAgent.piece)):
                        reward = -1000
                    else:
                        reward = 1

                    DQNAgent.DQNClass.storeTransition(currentState, actionIndex, reward, nextState)

                    if step > DQNAgent.DQNClass.targetUpdateRounds and step % DQNAgent.DQNClass.batchSize == 0:
                        DQNAgent.DQNClass.learn()
                step += 1
            
            if self.gameState.isWin(DQNAgent.piece): DQNWinningNum += 1
            if episode % 100 == 99:
                print("Episode:", episode + 1, "Winning rate:", DQNWinningNum / 100)
                winnings.append(DQNWinningNum / 100)
                if DQNWinningNum > 80:
                    DQNWinningGood += 1
                    if DQNWinningGood > 10:
                        plt.plot([i * 100 for i in range(int(trainingRounds / 100))], winnings)
                        plt.xlabel('epoch')
                        plt.ylabel('winning rate')
                        plt.title('8*8 chessboard, MCTS 0.001s')
                        plt.savefig('./lib/image/plot.png', dpi=600)
                        plt.show()
                        Helper.MESSAGE("Training Finished...")
                        torch.save(DQNAgent.DQNClass.evaluationNet, DQN_MODEL_PATH)
                        return True
                else:
                    DQNWinningGood = 0
                
                DQNWinningNum= 0
        plt.plot([i * 100 for i in range(int(trainingRounds / 100))], winnings)
        plt.xlabel('epoch')
        plt.ylabel('winning rate')
        plt.title('8*8 chessboard, MCTS 0.001s')
        plt.savefig('./lib/image/plot.png', dpi=600)
        plt.show()
        Helper.MESSAGE("Training Finished...")
        torch.save(DQNAgent.DQNClass.evaluationNet, DQN_MODEL_PATH)
        return True