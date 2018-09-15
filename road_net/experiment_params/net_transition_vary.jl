cond1 = make_condition_dict(num_nets=750,mcts_d=[8],n_bnds=[8,120],seed=5,
                            role=:trusted)
cond2 = make_condition_dict(num_nets=250,mcts_d=[3],n_bnds=[8,120],seed=4,
                            role=:candidate)
cond3 = make_condition_dict(num_nets=250,mcts_d=[1],n_bnds=[8,120],seed=3,
                            role=:candidate)

exp_dict = Dict(:name=>"net_transition_vary", :conditions=>Dict(1=>cond1, 2=>cond2, 3=> cond3))

fldr = "logs"
for x in keys(exp_dict[:conditions])
    fname = make_useful_fname(exp_dict,x)
    exp_dict[:conditions][x][:fname] = fname
    exp_dict[:conditions][x][:fldr] = fldr
end
