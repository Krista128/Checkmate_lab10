import chess

def evaluate_sunfish_style(fen):
    board = chess.Board(fen)
    
    # Ценности фигур
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    # Позиционные бонусы для пешек (центр важнее)
    pawn_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 15, 20, 20, 15, 10,  5,
        4,  8, 12, 16, 16, 12,  8,  4,
        3,  6,  9, 12, 12,  9,  6,  3,
        2,  4,  6,  8,  8,  6,  4,  2,
        1,  2,  3,  4,  4,  3,  2,  1,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            
            # Добавляем позиционный бонус для пешек
            if piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    value += pawn_table[square]
                else:
                    # Чёрные пешки — зеркально
                    value += pawn_table[63 - square]
            
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    return score