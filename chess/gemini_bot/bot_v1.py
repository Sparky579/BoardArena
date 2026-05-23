import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

def evaluate_board(board):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999
        else:
            return 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    if board.is_game_over():
        return evaluate_board(board)
    if depth == 0:
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -1000000
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 1000000
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def choose_action(state):
    board = chess.Board(state["fen"])
    best_move = None
    
    is_white = board.turn == chess.WHITE
    depth = 3
    
    if is_white:
        max_eval = -1000000
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, -1000000, 1000000, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
    else:
        min_eval = 1000000
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, -1000000, 1000000, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
                
    if best_move:
        return best_move.uci()
    return state["legal_actions"][0]
