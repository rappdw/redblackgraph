def topological_visit(g, v, color, order):
    """Run iterative DFS from node V"""
    total = 0
    stack = [v]  # create stack with starting vertex, stack to replace recursion with loop
    while stack:  # while stack is not empty
        v = stack[-1]  # peek top of stack
        if color[v]:  # if already seen
            v = stack.pop()  # done with this node, pop it from stack
            if color[v] == 1:  # if GRAY, finish this node
                order.append(v)
                color[v] = 2  # BLACK, done
        else:  # seen for first time
            color[v] = 1  # GRAY: discovered
            total += 1
            for w in range(len(g[0])): # for all neighbor (v, w)
                if w != v and g[v][w] and not color[w]:
                    stack.append(w)
    return total

def topological_sort(g):
    """Run DFS on graph"""
    color = [0] * len(g)
    order = [] # stack to hold topological ordering of graph
    for v in range(len(g)):
        if not color[v]:
            topological_visit(g, v, color, order)
    return order[::-1]


if __name__ == '__main__':
    A = [[-1, 2, 3, 0, 0],
         [ 0,-1, 0, 2, 0],
         [ 0, 0, 1, 0, 0],
         [ 0, 0, 0,-1, 0],
         [ 2, 0, 0, 0, 1]]

    print(topological_sort(A))