import chess
import time

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

PST = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
}

class ChessAI:
    def __init__(self):
        self.tt = {} 
        self.killer_moves = [[None] * 128 for _ in range(20)]
        self.start_time = 0
        self.time_limit = 1.7 

    def evaluate(self, board):
        if board.is_checkmate():
            return -999999 if board.turn == chess.WHITE else 999999
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            return 0

        score = 0
        piece_map = board.piece_map()
        
        for square, piece in piece_map.items():
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value + PST[piece.piece_type][chess.square_mirror(square)]
            else:
                score -= value + PST[piece.piece_type][square]

        white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
        black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
        if len(white_bishops) >= 2: score += 50
        if len(black_bishops) >= 2: score -= 50

        return score

    def move_priority(self, board, move, depth, best_move=None):
        if move == best_move:
            return 100000
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                return 1000 + PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type] // 10
            return 1000
        if depth < 20 and move == self.killer_moves[depth][0]:
            return 500
        if depth < 20 and move == self.killer_moves[depth][1]:
            return 400
        return 0

    def quiescence_search(self, board, alpha, beta, depth=0):
        if depth > 4:
            return self.evaluate(board)
            
        if depth % 10 == 0 and time.time() - self.start_time > self.time_limit:
            raise TimeoutError()

        stand_pat = self.evaluate(board)
        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if beta > stand_pat:
                beta = stand_pat

        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = self.quiescence_search(board, alpha, beta, depth + 1)
                board.pop()

                if board.turn == chess.WHITE:
                    if score >= beta:
                        return beta
                    if score > alpha:
                        alpha = score
                else:
                    if score <= alpha:
                        return alpha
                    if score < beta:
                        beta = score
        return alpha if board.turn == chess.WHITE else beta

    def minimax(self, board, depth, alpha, beta, maximizing_player, ply=0):
        if ply % 10 == 0 and time.time() - self.start_time > self.time_limit:
            raise TimeoutError()

        board_hash = board.board_fen() 
        if board_hash in self.tt:
            tt_depth, tt_score = self.tt[board_hash]
            if tt_depth >= depth:
                return tt_score

        if board.is_game_over():
            return self.evaluate(board)
        
        if depth == 0:
            return self.quiescence_search(board, alpha, beta)

        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda m: self.move_priority(board, m, depth), reverse=True)

        if maximizing_player:
            max_eval = -10000000
            for move in legal_moves:
                board.push(move)
                try:
                    eval = self.minimax(board, depth - 1, alpha, beta, False, ply + 1)
                except TimeoutError:
                    board.pop()
                    raise
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                alpha = max(alpha, eval)
                if beta <= alpha:
                    if not board.is_capture(move) and depth < 20:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    break
            if depth > 1:
                self.tt[board_hash] = (depth, max_eval)
            return max_eval
        else:
            min_eval = 10000000
            for move in legal_moves:
                board.push(move)
                try:
                    eval = self.minimax(board, depth - 1, alpha, beta, True, ply + 1)
                except TimeoutError:
                    board.pop()
                    raise
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                beta = min(beta, eval)
                if beta <= alpha:
                    if not board.is_capture(move) and depth < 20:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    break
            if depth > 1:
                self.tt[board_hash] = (depth, min_eval)
            return min_eval

    def choose_action(self, state):
        board = chess.Board(state["fen"])
        self.start_time = time.time()
        best_move = None
        
        try:
            for d in range(1, 7): 
                current_best_move = None
                legal_moves = list(board.legal_moves)
                legal_moves.sort(key=lambda m: self.move_priority(board, m, d, best_move), reverse=True)
                
                if board.turn == chess.WHITE:
                    max_eval = -10000000
                    for move in legal_moves:
                        board.push(move)
                        eval = self.minimax(board, d - 1, -10000000, 10000000, False, 1)
                        board.pop()
                        if eval > max_eval:
                            max_eval = eval
                            current_best_move = move
                else:
                    min_eval = 10000000
                    for move in legal_moves:
                        board.push(move)
                        eval = self.minimax(board, d - 1, -10000000, 10000000, True, 1)
                        board.pop()
                        if eval < min_eval:
                            min_eval = eval
                            current_best_move = move
                best_move = current_best_move
        except TimeoutError:
            pass
            
        if best_move:
            return best_move.uci()
        return state["legal_actions"][0]

def choose_action(state):
    ai = ChessAI()
    return ai.choose_action(state)
