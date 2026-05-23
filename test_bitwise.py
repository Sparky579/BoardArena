import time

def bfs_path(pos, target_row, h_mask, v_mask):
    q = [0] * 81
    q[0] = pos
    visited = 1 << pos
    dist = 0
    parent = [-1] * 81
    
    head = 0
    tail = 1
    
    while head < tail:
        level_tail = tail
        for i in range(head, level_tail):
            curr = q[i]
            r = curr // 9
            if r == target_row:
                path = []
                p = curr
                while p != -1:
                    path.append(p)
                    p = parent[p]
                path.reverse()
                return dist, path
                
            if r > 0:
                nxt = curr - 9
                if not (h_mask & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            if r < 8:
                nxt = curr + 9
                if not (h_mask & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            c = curr % 9
            if c > 0:
                nxt = curr - 1
                if not (v_mask & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            if c < 8:
                nxt = curr + 1
                if not (v_mask & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
        head = level_tail
        dist += 1
        
    return 1000, []

t0 = time.perf_counter()
for _ in range(10000):
    bfs_path(76, 0, 0, 0)
t1 = time.perf_counter()
print(f"10000 old_bfs took {t1-t0:.4f}s")
