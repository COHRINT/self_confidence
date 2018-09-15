cond1 = make_condition_dict(num_nets=5,mcts_d=[8],seed=5,role=:trusted)
cond2 = make_condition_dict(num_nets=250,mcts_d=[3],seed=4,role=:candidate)
cond3 = make_condition_dict(num_nets=250,mcts_d=[1],seed=3,role=:candidate)

exp_dict = Dict(:name=>"x4_test", :conditions=>Dict(1=>cond1, 2=>cond2, 3=> cond3))

fldr = "logs"
for x in keys(exp_dict[:conditions])
    fname = make_useful_fname(exp_dict,x)
    exp_dict[:conditions][x][:fname] = fname
    exp_dict[:conditions][x][:fldr] = fldr
end
