function my_rem_vertex!(g,n)
    # removes the edges, and leaves the vertex hanging alone in space. This helps preserve node names
    for edg in full(adjacency_matrix(g))[:,n]
        edg = 0
    end
    for edg in full(adjacency_matrix(g))[n,:]
        edg = 0
    end
    return g
end
