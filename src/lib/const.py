import sys
from enum import Enum, unique
sys.path.append ("..")
sys.path.append ('.')


DEFAULT_EXIT_PREFIX = ">> [ EXIT ] "
DEFAULT_PANIC_PREFIX = ">> [ PANIC ] "
DEFAULT_GUI_PREFIX = ">> [ GUI ] "
DEFAULT_WARNING_PREFIX = ">> [ WARNING ] "
DEFAULT_WIN_REFIX = ">> [ WIN ] "
DEFAULT_MSG_REFIX = ">> [ MSG ] "
CHESS_ONE_SOUND = "./data/sound/chessone.wav"
WINDOW_WIDTH = 760
WINDOW_HEIGHT = 760
DRAW_LEFT_UPON = (100, 100)
DRAW_RIGHT_DOWN = (660, 660)
GRID_NUM = 7
PIECE_SIZE = 10

DQN_MODEL_PATH = './lib/model/DQN/model.pt'
DQN_Random_Trained_MODEL_PATH = './lib/model/DQN/trainedByRandom.pt'
DQN_MCTS_88_Trained_MODEL_PATH = './lib/model/DQN/trainedByMCTS_time_0.001_10000.pt'

# 2020/12/5 Limzh -- Add 
# Enumeration class: Piece
# When grids[i, j] has no piece, its value should be `None`;
# Black should be Piece.BLACK, white should be Piece.WHITE.
@unique
class Piece (Enum):
    BLACK = 'Black'
    WHITE = 'White'

if __name__ == "__main__":
    print (Piece.BLACK, type (Piece.BLACK))



