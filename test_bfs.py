import collections
import time

def get_shortest_path(pos, goal_row, h_edges, v_edges):
    q = collections.deque([pos])
    visited = 1 << pos
    dist = 0
    parent = {pos: -1}
    
    while q:
        for _ in range(len(q)):
            curr = q.popleft()
            r = curr // 9
            if r == goal_row:
                path = []
                p = curr
                while p != -1:
                    path.append(p)
                    p = parent[p]
                path.reverse()
                return dist, path
            
            if r > 0:
                nxt = curr - 9
                if not (h_edges & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q.append(nxt)
            if r < 8:
                nxt = curr + 9
                if not (h_edges & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q.append(nxt)
            c = curr % 9
            if c > 0:
                nxt = curr - 1
                if not (v_edges & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q.append(nxt)
            if c < 8:
                nxt = curr + 1
                if not (v_edges & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q.append(nxt)
        dist += 1
    return 1000, []

def is_wall_valid(h_walls, v_walls, dir, r, c):
    if dir == 'H':
        # check bounds
        if r < 0 or r > 7 or c < 0 or c > 7: return False
        # check overlap
        if h_walls & (1 << (r * 9 + c)): return False
        if h_walls & (1 << (r * 9 + c + 1)): return False # shouldn't happen independently but just in case
        if h_walls & (1 << (r * 9 + c - 1)) if c > 0 else False: return False # overlapping H walls
        # The actual H wall overlap logic:
        # H wall at r,c overlaps with H wall at r,c-1 and r,c+1 (but they share a cell, so they overlap if they share the center)
        pass
    return True

print("Test BFS")
dist, path = get_shortest_path(4, 8, 0, 0)
print(dist, path)
