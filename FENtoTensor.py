import chess 
import numpy as np


def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            tensor[row, col, channel] = 1.0
    
    return tensor
    
def result_to_label(result):
    if result == '0-1':  # Проигрыш белых
        return 1  # Была ошибка
    elif result == '1-0':  # Выигрыш белых
        return 0  # Не было ошибок
    else:  # Ничья
        return 0

def extract_fen_pairs(fen_list):
    pairs = []
    for i in range(len(fen_list) - 1):
        pairs.append((fen_list[i], fen_list[i+1]))
    return pairs