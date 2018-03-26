# function plot_mg(mg::my_net_graph;v_names=vertices(mg),f_name="test")
    # plot_mg(mg.g,v_names=v_names,f_name=f_name)
# end
function plot_mg(g::MetaGraphs.MetaGraph;v_names=vertices(g),f_name="test")
    # labels = [string(x[1],"\\\\xx.x") for x in g.vprops if !isempty(x)]
    labels = [string(x[1]) for x in g.vprops if !isempty(x)]

    # tp = TikzGraphs.plot(g.graph,Layouts.Spring(randomSeed=10), labels=labels, node_style="align=center, draw, very thick, rounded corners, fill=blue!10", edge_style="very thick, sibling distance=19cm")
    tp = TikzGraphs.plot(g.graph,Layouts.Spring(randomSeed=10), node_style="draw, fill=blue!10, node distance=28cm", edge_style="sibling distance=19cm")

    TikzPictures.save(PDF(f_name), tp)
end
