"""Graph cut / min-cut max-flow (brief Item 4). Owner: P5.

Contract (see README "Shared interface"):
    find_vertical_seam(energy) -> seam
    energy: H x W float array, all values >= 0
    seam:   int array of shape (H,), seam[i] = column removed in row i,
            with |seam[i] - seam[i-1]| <= 1

Test without the rest of the pipeline, e.g.:
    energy = np.random.default_rng(0).random((20, 30))

------------------------------------------------------------------------
BIG PICTURE (water-pipe picture)
------------------------------------------------------------------------
* Every pixel (i, j) becomes a NODE in a graph.
* We add two extra nodes: the SOURCE s (a tap) and the SINK t (a drain).
* EDGES are one-way pipes. Each pipe has a CAPACITY (how much water fits).

Pipes we build, for a vertical seam:
  1. s -> every pixel in column 0                       capacity INF
  2. pixel (i, j) -> its right neighbour (i, j+1)       capacity e[i, j]
     (snipping this pipe == "remove pixel (i, j)")
  3. last-column pixel (i, W-1) -> t                    capacity e[i, W-1]
     (t acts like an extra column on the right, so the last column can also
      be removed, and a 1-column image does not create an INF path s -> t)
  4. "rule" pipes, all capacity INF, so a cut that breaks the seam rules
     would cost infinity and can never be the minimum:
       - backward  (i, j+1) -> (i, j)         => exactly one cut per row
       - diagonal  (i, j) -> (i-1, j-1)       => adjacent rows' cuts are
       - diagonal  (i, j) -> (i+1, j-1)          at most 1 column apart

Then:
  A. Edmonds-Karp: repeatedly find the shortest s -> t path with spare room
     (BFS), push the bottleneck amount of water along it, until no path is left.
  B. One last BFS from s through pipes that still have room. The nodes it
     reaches are the SOURCE SIDE S of the minimum cut.
  C. In each row, the rightmost pixel in S is the pixel whose rightward pipe
     is full (cut). That pixel is the seam pixel.

Max-flow min-cut theorem: total water pushed == capacity of the minimum cut
== total energy of the seam, so the seam is the cheapest possible one
(the same cost DP finds).
"""

from collections import deque

import numpy as np

INF = float("inf")

# Spare room smaller than this is treated as "full". Avoids endless tiny
# augmentations caused by floating-point rounding (e.g. 0.3 - 0.1 - 0.2 != 0).
EPS = 1e-12


def _build_graph(energy):
    """Build the residual graph for a vertical seam.

    Returns (room, s, t):
      room[u] is a dict {v: spare capacity of pipe u -> v}.
      Nodes are plain integers:
        pixel (i, j) -> i * W + j
        source s     -> H * W
        sink t       -> H * W + 1
    """
    H, W = energy.shape
    s = H * W
    t = H * W + 1

    # One dict per node. room[u][v] = how much more water pipe u -> v can take.
    room = [dict() for _ in range(H * W + 2)]

    def add_edge(u, v, capacity):
        # Add capacity to pipe u -> v (+= in case the pipe already exists).
        room[u][v] = room[u].get(v, 0.0) + capacity
        # Make sure the REVERSE pipe v -> u exists too (with 0 room if new).
        # Edmonds-Karp uses it to "undo" water pushed earlier.
        room[v].setdefault(u, 0.0)

    def node(i, j):
        # Turn a (row, column) pixel position into a single node number.
        return i * W + j

    for i in range(H):
        # 1. Tap feeds every pixel in the first column.
        add_edge(s, node(i, 0), INF)

        for j in range(W):
            if j + 1 < W:
                # 2. Rightward pipe. Its capacity is the cost of removing (i, j).
                add_edge(node(i, j), node(i, j + 1), float(energy[i, j]))
                # 4a. Backward INF pipe: stops a row from being cut twice.
                add_edge(node(i, j + 1), node(i, j), INF)
            else:
                # 3. Last column drains into t (t is a virtual extra column).
                add_edge(node(i, j), t, float(energy[i, j]))

            # 4b. Diagonal INF pipes going one column LEFT and one row up/down.
            #     They stop neighbouring rows' seam pixels from being more
            #     than 1 column apart (keeps the seam connected).
            if j > 0:
                if i > 0:
                    add_edge(node(i, j), node(i - 1, j - 1), INF)
                if i + 1 < H:
                    add_edge(node(i, j), node(i + 1, j - 1), INF)

    return room, s, t


def _bfs(room, s, t=None):
    """Breadth-first search from s, only through pipes with spare room.

    Returns `parent`, a dict {node: the node we came from}. Every node that is
    a key in `parent` was reached. If t is given, we stop as soon as we reach it.
    """
    parent = {s: None}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v, spare in room[u].items():
            if spare > EPS and v not in parent:
                parent[v] = u
                if v == t:
                    return parent  # found the sink, the path is complete
                queue.append(v)
    return parent


def _max_flow(room, s, t):
    """Stage A: Edmonds-Karp. Pushes water until no s -> t path has room.

    Modifies `room` in place (it becomes the final residual graph) and
    returns the total flow pushed.
    """
    total_flow = 0.0
    while True:
        # Shortest path (fewest pipes) from s to t that still has room.
        parent = _bfs(room, s, t)
        if t not in parent:
            break  # no path left: the flow is maximal

        # Walk back from t to s to find the bottleneck (smallest spare room).
        bottleneck = INF
        v = t
        while v != s:
            u = parent[v]
            bottleneck = min(bottleneck, room[u][v])
            v = u

        # Push `bottleneck` units of water along the path:
        #   forward pipe u -> v loses room,
        #   reverse pipe v -> u gains room (so it can be undone later).
        v = t
        while v != s:
            u = parent[v]
            room[u][v] -= bottleneck
            room[v][u] += bottleneck
            v = u

        total_flow += bottleneck
    return total_flow


def find_vertical_seam(energy):
    energy = np.asarray(energy, dtype=float)
    H, W = energy.shape

    room, s, t = _build_graph(energy)

    # Stage A: push as much water as possible from s to t.
    _max_flow(room, s, t)

    # Stage B: every node still reachable from s is on the source side S.
    source_side = _bfs(room, s)  # dict keys = reachable nodes

    # Stage C: in each row, the rightmost pixel in S is the seam pixel.
    # (Column 0 is always in S because s -> (i, 0) has INF room.)
    seam = np.zeros(H, dtype=int)
    for i in range(H):
        rightmost = 0
        for j in range(W):
            if i * W + j in source_side:
                rightmost = j
        seam[i] = rightmost
    return seam
