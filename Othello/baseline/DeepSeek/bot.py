import time

FULL_MASK = 0xFFFFFFFFFFFFFFFF
A_FILE = 0x0101010101010101
H_FILE = 0x8080808080808080
NOT_A_FILE = FULL_MASK ^ A_FILE
NOT_H_FILE = FULL_MASK ^ H_FILE

CORNER_INDICES = [0, 7, 56, 63]
X_SQUARE_INDICES = {9, 14, 49, 54}

_ADJ_TO_CORNER = {}
for _ci, _adj_list in [(0, [1, 8, 9]), (7, [6, 15, 14]),
                        (56, [48, 57, 49]), (63, [55, 62, 54])]:
    for _a in _adj_list:
        _ADJ_TO_CORNER[_a] = _ci

_EDGE_H = {
    0:  list(range(0, 8)),
    7:  list(range(7, -1, -1)),
    56: list(range(56, 64)),
    63: list(range(63, 55, -1)),
}
_EDGE_V = {
    0:  list(range(0, 64, 8)),
    7:  list(range(7, 64, 8)),
    56: list(range(56, -1, -8)),
    63: list(range(63, -1, -8)),
}

_NEIGHBORS = [[] for _ in range(64)]
for _idx in range(64):
    _r, _c = _idx // 8, _idx % 8
    for _dr in (-1, 0, 1):
        for _dc in (-1, 0, 1):
            if _dr == 0 and _dc == 0:
                continue
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < 8 and 0 <= _nc < 8:
                _NEIGHBORS[_idx].append(_nr * 8 + _nc)

POSITION_WEIGHTS = [
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,   0,   0,  -1,  -2,   5,
      5,  -2,  -1,   0,   0,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
]


def _popcount(x):
    return x.bit_count()


def _bit_to_square(bit):
    idx = bit.bit_length() - 1
    return chr(ord('a') + idx % 8) + str(idx // 8 + 1)


def _bits_to_list(bits):
    result = []
    while bits:
        lsb = bits & -bits
        result.append(lsb)
        bits ^= lsb
    return result


def _shift_n(b):
    return (b << 8) & FULL_MASK

def _shift_s(b):
    return b >> 8

def _shift_e(b):
    return (b & NOT_H_FILE) << 1

def _shift_w(b):
    return (b & NOT_A_FILE) >> 1

def _shift_ne(b):
    return ((b & NOT_H_FILE) << 9) & FULL_MASK

def _shift_nw(b):
    return ((b & NOT_A_FILE) << 7) & FULL_MASK

def _shift_se(b):
    return (b & NOT_H_FILE) >> 7

def _shift_sw(b):
    return (b & NOT_A_FILE) >> 9


_DIRECTIONS = [_shift_n, _shift_s, _shift_e, _shift_w,
               _shift_ne, _shift_nw, _shift_se, _shift_sw]

_OPPOSITE = {
    _shift_n: _shift_s, _shift_s: _shift_n,
    _shift_e: _shift_w, _shift_w: _shift_e,
    _shift_ne: _shift_sw, _shift_sw: _shift_ne,
    _shift_nw: _shift_se, _shift_se: _shift_nw,
}

_DIR_PAIRS = [(d, _OPPOSITE[d]) for d in _DIRECTIONS]
_CORNER_BITS = [1 << ci for ci in CORNER_INDICES]


def _generate_moves_flips(p, o):
    empty = ~(p | o) & FULL_MASK
    result = {}
    for sf, rev in _DIR_PAIRS:
        flanked = sf(p) & o
        for _ in range(5):
            flanked |= sf(flanked) & o
        candidates = sf(flanked) & empty
        while candidates:
            lsb = candidates & -candidates
            candidates ^= lsb
            flip_mask = 0
            bits = rev(lsb)
            while bits & o:
                flip_mask |= bits
                bits = rev(bits)
            if bits & p:
                existing = result.get(lsb, 0)
                result[lsb] = existing | flip_mask
    return [(m, f) for m, f in result.items()]


def _mobility_count(p, o):
    empty = ~(p | o) & FULL_MASK
    moves = 0
    for sf, _ in _DIR_PAIRS:
        flanked = sf(p) & o
        for _ in range(5):
            flanked |= sf(flanked) & o
        moves |= sf(flanked) & empty
    return _popcount(moves)


def _edge_stability(p, stable_mask):
    new_stable = 0
    for ci in CORNER_INDICES:
        cb = 1 << ci
        if stable_mask & cb:
            for idx in _EDGE_H[ci]:
                bit = 1 << idx
                if stable_mask & bit:
                    continue
                if p & bit:
                    new_stable |= bit
                    stable_mask |= bit
                else:
                    break
            for idx in _EDGE_V[ci]:
                bit = 1 << idx
                if stable_mask & bit:
                    continue
                if p & bit:
                    new_stable |= bit
                    stable_mask |= bit
                else:
                    break
    return new_stable


def _compute_stable(p, o):
    stable = 0
    for cb in _CORNER_BITS:
        if p & cb:
            stable |= cb
    while True:
        added = _edge_stability(p, stable)
        if not added:
            break
        stable |= added
    return stable


class Bot:
    name = "DeepSeek"

    def __init__(self):
        self.tt = {}
        self.killer_moves = [[0, 0] for _ in range(64)]
        self.history = [0] * 64
        self.deadline = 0.0
        self.timeout = False
        self.node_count = 0

    def _state_to_bitboards(self, state):
        board = state["board"]
        black = 0
        white = 0
        for r in range(8):
            row = board[r]
            for c in range(8):
                ch = row[c]
                if ch == '.':
                    continue
                bit = 1 << ((7 - r) * 8 + c)
                if ch == 'B':
                    black |= bit
                elif ch == 'W':
                    white |= bit
        return black, white

    def _check_time(self):
        self.node_count += 1
        if self.node_count & 255 == 0:
            if time.time() >= self.deadline:
                self.timeout = True
        return self.timeout

    def choose_action(self, state):
        legal = state["legal_actions"]
        if legal == ["PASS"]:
            return "PASS"

        black, white = self._state_to_bitboards(state)
        player_id = state["player_id"]
        if player_id == 0:
            p, o = black, white
        else:
            p, o = white, black

        self.start_time = time.time()
        time_limit = state.get("decision_timeout", 1.5)
        self.deadline = self.start_time + time_limit * 0.9
        self.timeout = False
        self.node_count = 0
        self.tt.clear()
        self.killer_moves = [[0, 0] for _ in range(64)]
        self.history = [0] * 64

        empty_count = _popcount(~(p | o) & FULL_MASK)
        best_move_str = legal[0]
        prev_score = None

        for depth in range(1, 64):
            if self.timeout:
                break

            if prev_score is not None and depth >= 3:
                window = 80
                alpha = prev_score - window
                beta = prev_score + window
                score, move_bit = self._root_search(p, o, depth, alpha, beta)
                if self.timeout:
                    break
                if score <= alpha or score >= beta:
                    score, move_bit = self._root_search(p, o, depth, -float('inf'), float('inf'))
            else:
                score, move_bit = self._root_search(p, o, depth, -float('inf'), float('inf'))

            if self.timeout:
                break
            if move_bit != 0:
                best_move_str = _bit_to_square(move_bit)
                prev_score = score
                if score > 9000 or (empty_count <= depth and empty_count <= 14):
                    break

        return best_move_str

    def _root_search(self, p, o, depth, alpha, beta):
        moves = _generate_moves_flips(p, o)
        if not moves:
            return -float('inf'), 0

        self._order_moves(moves, p, o, depth, 0)

        best_score = -float('inf')
        best_move = moves[0][0]

        for i, (move_bit, flip_mask) in enumerate(moves):
            if self.timeout:
                return best_score, best_move

            new_p = p | move_bit | flip_mask
            new_o = o ^ flip_mask
            idx = move_bit.bit_length() - 1

            if i == 0:
                score = -self._negamax(new_o, new_p, depth - 1, -beta, -alpha)
            else:
                score = -self._negamax(new_o, new_p, depth - 1, -alpha - 1, -alpha)
                if not self.timeout and alpha < score < beta:
                    score = -self._negamax(new_o, new_p, depth - 1, -beta, -alpha)

            if self.timeout:
                return best_score, best_move

            if score > best_score:
                best_score = score
                best_move = move_bit
                if score > alpha:
                    alpha = score
                if score >= beta:
                    killers = self.killer_moves[depth]
                    if move_bit != killers[0]:
                        killers[1] = killers[0]
                        killers[0] = move_bit
                    self.history[idx] += depth * depth
                    break

        return best_score, best_move

    def _negamax(self, p, o, depth, alpha, beta):
        if self.timeout:
            return 0

        self.node_count += 1
        if self.node_count & 255 == 0 and time.time() >= self.deadline:
            self.timeout = True
            return 0

        empty = ~(p | o) & FULL_MASK
        empty_count = _popcount(empty)

        if empty_count == 0:
            return self._terminal_score(p, o)

        tt_key = (p, o)
        tt_entry = self.tt.get(tt_key)
        tt_move = 0
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == 0:
                    return tt_score
                elif tt_flag == 1:
                    if tt_score >= beta:
                        return tt_score
                    alpha = max(alpha, tt_score)
                elif tt_flag == 2:
                    if tt_score <= alpha:
                        return tt_score
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        moves = _generate_moves_flips(p, o)

        if not moves:
            opp_moves = _generate_moves_flips(o, p)
            if not opp_moves:
                return self._terminal_score(p, o)
            return -self._negamax(o, p, depth, -beta, -alpha)

        if depth <= 0:
            return self._evaluate(p, o, empty_count)

        self._order_moves(moves, p, o, depth, tt_move)

        best_score = -float('inf')
        best_move = 0
        flag = 2

        for i, (move_bit, flip_mask) in enumerate(moves):
            if self.timeout:
                return best_score if best_score > -float('inf') else 0

            new_p = p | move_bit | flip_mask
            new_o = o ^ flip_mask
            idx = move_bit.bit_length() - 1

            if i == 0:
                score = -self._negamax(new_o, new_p, depth - 1, -beta, -alpha)
            else:
                do_lmr = (i >= 4 and depth >= 3 and (idx not in CORNER_INDICES) and
                          move_bit != tt_move and move_bit != self.killer_moves[depth][0] and
                          move_bit != self.killer_moves[depth][1] and alpha > -9999)

                if do_lmr:
                    r = 1 if depth < 6 else (2 + (i - 4) // 6)
                    lmr_depth = depth - 1 - r
                    if lmr_depth > 0:
                        score = -self._negamax(new_o, new_p, lmr_depth, -alpha - 1, -alpha)
                    else:
                        score = -self._evaluate(new_o, new_p,
                                                _popcount(~(new_o | new_p) & FULL_MASK))
                    if score <= alpha:
                        continue
                    score = -self._negamax(new_o, new_p, depth - 1, -beta, -alpha)
                else:
                    score = -self._negamax(new_o, new_p, depth - 1, -alpha - 1, -alpha)
                    if not self.timeout and alpha < score < beta:
                        score = -self._negamax(new_o, new_p, depth - 1, -beta, -alpha)

            if self.timeout:
                return best_score if best_score > -float('inf') else 0

            if score > best_score:
                best_score = score
                best_move = move_bit
                if score > alpha:
                    alpha = score
                    flag = 0
                if score >= beta:
                    flag = 1
                    killers = self.killer_moves[depth]
                    if move_bit != killers[0]:
                        killers[1] = killers[0]
                        killers[0] = move_bit
                    self.history[idx] += depth * depth
                    break

        if not self.timeout:
            self.tt[tt_key] = (depth, best_score, flag, best_move)

        return best_score

    def _order_moves(self, moves, p, o, depth, tt_move):
        empty = ~(p | o) & FULL_MASK
        killer0 = self.killer_moves[depth][0]
        killer1 = self.killer_moves[depth][1]

        scored = []
        for move_bit, flip_mask in moves:
            score = 0
            if move_bit == tt_move:
                score = 1_000_000
            elif move_bit == killer0:
                score = 100_000
            elif move_bit == killer1:
                score = 99_000

            idx = move_bit.bit_length() - 1
            score += self.history[idx]

            if idx in CORNER_INDICES:
                score += 50_000
            elif idx in _ADJ_TO_CORNER:
                corner_idx = _ADJ_TO_CORNER[idx]
                corner_bit = 1 << corner_idx
                if corner_bit & empty:
                    if idx in X_SQUARE_INDICES:
                        score -= 40_000
                    else:
                        score -= 20_000
                else:
                    if corner_bit & p:
                        score += 5_000

            scored.append((score, move_bit, flip_mask))

        scored.sort(key=lambda x: x[0], reverse=True)
        moves[:] = [(m, f) for _, m, f in scored]

    def _evaluate(self, p, o, empty_count):
        if empty_count <= 14:
            return _popcount(p) - _popcount(o)

        score = 0
        empty = ~(p | o) & FULL_MASK

        p_stable = _compute_stable(p, o)
        o_stable = _compute_stable(o, p)

        p_frontier = 0
        o_frontier = 0

        for idx in range(64):
            bit = 1 << idx
            if p & bit:
                score += POSITION_WEIGHTS[idx]
                if not (p_stable & bit):
                    for nb_idx in _NEIGHBORS[idx]:
                        if empty & (1 << nb_idx):
                            p_frontier += 1
                            break
            elif o & bit:
                score -= POSITION_WEIGHTS[idx]
                if not (o_stable & bit):
                    for nb_idx in _NEIGHBORS[idx]:
                        if empty & (1 << nb_idx):
                            o_frontier += 1
                            break

        p_stable_count = _popcount(p_stable)
        o_stable_count = _popcount(o_stable)
        score += p_stable_count * 20
        score -= o_stable_count * 20

        p_total = _popcount(p)
        o_total = _popcount(o)
        p_interior = p_total - p_frontier - p_stable_count
        o_interior = o_total - o_frontier - o_stable_count
        if p_interior > 0:
            score += p_interior * 8
        if o_interior > 0:
            score -= o_interior * 8

        for ci in CORNER_INDICES:
            corner_bit = 1 << ci
            if p & corner_bit:
                score += 110
            elif o & corner_bit:
                score -= 110
            elif corner_bit & empty:
                for adj_idx in _ADJ_TO_CORNER:
                    if _ADJ_TO_CORNER[adj_idx] == ci:
                        adj_bit = 1 << adj_idx
                        if p & adj_bit:
                            score -= 55 if adj_idx in X_SQUARE_INDICES else 30
                        elif o & adj_bit:
                            score += 55 if adj_idx in X_SQUARE_INDICES else 30

        if empty_count > 40:
            score -= (p_frontier - o_frontier) * 3
        elif empty_count > 20:
            score -= (p_frontier - o_frontier) * 6
        else:
            score -= (p_frontier - o_frontier) * 10

        if empty_count <= 35:
            p_mob = _mobility_count(p, o)
            o_mob = _mobility_count(o, p)
            mob = p_mob - o_mob
            if empty_count > 25:
                score += mob * 5
            elif empty_count > 16:
                score += mob * 3
            else:
                score += mob * 2

        if 5 <= empty_count <= 20:
            if empty_count % 2 == 1:
                score += 5

        return score

    def _terminal_score(self, p, o):
        diff = _popcount(p) - _popcount(o)
        if diff > 0:
            return 10000 + diff
        elif diff < 0:
            return -10000 + diff
        return 0
