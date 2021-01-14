import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import csv

from lib.const import *

class DQNClass:
    def __init__(self, boardWidth=15, boardHeight=15, learningRate=0.1, discount=0.9, epsilon=0.1, memorySize=2000, targetUpdateRounds=200, batchSize=20, netType='CNN'):
        self.boardWidth = boardWidth
        self.boardHeight = boardHeight
        self.actionsNum = self.boardWidth * self.boardHeight
        self.learningRate = learningRate
        self.discount = discount
        self.netType = netType

        self.epsilon = epsilon
        self.epsilonDecent = 0.0001
        self.epsilonThreshold = 0.01

        self.memorySize = memorySize
        self.currentStateMemory = torch.zeros((self.memorySize, self.boardHeight, self.boardWidth))
        self.nextStateMemory = torch.zeros((self.memorySize, self.boardHeight, self.boardWidth))
        self.actionsAndRewardMemory = torch.zeros((self.memorySize, 2))
        self.memoryCounter = 0
        self.batchSize = batchSize

        self.learnCounter = 0
        self.targetUpdateRounds = targetUpdateRounds

        if self.netType == 'CNN':
            self.evaluationNet = DQN(self.boardWidth, self.boardHeight, self.actionsNum)
            self.targetNet = DQN(self.boardWidth, self.boardHeight, self.actionsNum)
        elif self.netType == 'FC':
            self.evaluationNet = DQN_FC(self.boardWidth, self.boardHeight, self.actionsNum)
            self.targetNet = DQN_FC(self.boardWidth, self.boardHeight, self.actionsNum)
        else:
            print("Invalid network type!")
            exit()

        self.optimizer = optim.SGD(self.evaluationNet.parameters(), lr=self.learningRate, momentum=0.9)

    def chooseAction(self, observation, successors):
        availableActions = [succ[0] for succ in successors]
        if self.netType == 'CNN':
            inputObservation = self.numpyToTensor(observation).unsqueeze(0).unsqueeze(0)
        elif self.netType == 'FC':
            inputObservation = self.boardToVector(self.numpyToTensor(observation)).unsqueeze(0)

        if np.random.uniform() > self.epsilon:
            with torch.no_grad():
                actionValues = np.squeeze(self.evaluationNet(inputObservation).numpy())
                while True:
                    actionIndex = np.argmax(actionValues)
                    action = (actionIndex // self.boardHeight, actionIndex % self.boardHeight)
                    if action in availableActions:
                        return action
                    else:
                        actionValues[actionIndex] = float('-inf')
        else:
            while True:
                actionIndex = np.random.randint(0, self.actionsNum)
                action = (actionIndex // self.boardHeight, actionIndex % self.boardHeight)
                if action in availableActions:
                    return action

    def storeTransition(self, state, action, reward, nextState):
        index = self.memoryCounter % self.memorySize
        self.currentStateMemory[index,:] = self.numpyToTensor(state)
        self.actionsAndRewardMemory[index, 0] = float(action)
        self.actionsAndRewardMemory[index, 1] = reward
        self.nextStateMemory[index,:] = self.numpyToTensor(nextState)
        self.memoryCounter += 1

    def learn(self):
        if self.learnCounter % self.targetUpdateRounds == 0:
            torch.save(self.evaluationNet, DQN_MODEL_PATH)
            self.targetNet = torch.load(DQN_MODEL_PATH)

        if self.memoryCounter > self.memorySize:
            sampleIndex = np.random.choice(self.memorySize, size=self.batchSize)
        else:
            sampleIndex = np.random.choice(self.memoryCounter, size=self.batchSize)
        
        if self.netType == 'CNN':
            currentStateBatch = self.currentStateMemory[sampleIndex, :].unsqueeze(1)
            nextStateBatch = self.nextStateMemory[sampleIndex,:].unsqueeze(1)
        elif self.netType == 'FC':
            currentStateBatch = torch.from_numpy(np.array([self.boardToVector(state).numpy() for state in self.currentStateMemory[sampleIndex, :]]))
            nextStateBatch = torch.from_numpy(np.array([self.boardToVector(state).numpy() for state in self.nextStateMemory[sampleIndex, :]]))

        actionsAndRewardBatch = self.actionsAndRewardMemory[sampleIndex,:]
        
        currentQValues = self.evaluationNet(currentStateBatch)
        nextQValues = self.targetNet(nextStateBatch)
        targetQValues = currentQValues.clone().detach()

        batchIndices = np.arange(self.batchSize, dtype=np.int32)
        actionIndices = actionsAndRewardBatch[:, 0].long()
        rewards = actionsAndRewardBatch[:, 1]

        targetQValues[batchIndices, actionIndices] = rewards + self.discount * nextQValues.max(1)[0].detach()

        loss = F.smooth_l1_loss(currentQValues, targetQValues)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.evaluationNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        if self.epsilon > self.epsilonThreshold: self.epsilon -= self.epsilonDecent
        self.learnCounter += 1
    
    def numpyToTensor(self, state):
        result = torch.zeros((self.boardHeight, self.boardWidth))
        for i in range(self.boardHeight):
            for j in range(self.boardWidth):
                if state[i, j] == Piece.BLACK:
                    result[i, j] = 1
                elif state[i, j] == Piece.WHITE:
                    result[i, j] = 2
                else:
                    result[i, j] = 0
        return result

    def boardToVector(self, state):
        result = torch.zeros(self.boardWidth * self.boardHeight)
        for i in range(self.boardWidth):
            for j in range(self.boardHeight):
                result[i * self.boardHeight + j] = state[j, i]

        return result

class DQN(nn.Module):
    def __init__(self, width, height, outputNum):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        def conv2dSizeOut(size, kernelSize=3, stride=1):
            return (size - (kernelSize - 1) - 1) // stride + 1
        
        convWidth = conv2dSizeOut(conv2dSizeOut(width, 3, 1), 3, 1)
        convHeight = conv2dSizeOut(conv2dSizeOut(height, 3, 1), 3, 1)
        self.head = nn.Linear(convWidth * convHeight * 32, outputNum)
        # self.head = nn.Linear(442, outputNum)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQN_FC(nn.Module):
    def __init__(self, width, height, outputNum):
        super(DQN_FC, self).__init__()
        self.fc1 = nn.Linear(width * height, 640)
        self.fc2 = nn.Linear(640, 400)
        self.fc3 = nn.Linear(400, 280)
        self.fc4 = nn.Linear(280, 180)
        self.fc5 = nn.Linear(180, 100)
        self.fc6 = nn.Linear(100, outputNum)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x