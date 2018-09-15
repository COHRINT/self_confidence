# vary transition prob, mcts_depth, and number of nodes in network
cond1 = make_condition_dict(num_nets=5,mcts_d=[10],n_bnds=[8,35],seed=5,repeats=25,steps=25,role=:trusted)
cond2 = make_condition_dict(num_nets=3,mcts_d=[1],n_bnds=[8,35],seed=4,repeats=25,steps=25)
cond3 = make_condition_dict(num_nets=2,mcts_d=[2],n_bnds=[8,35],seed=3,repeats=25,steps=25)
cond4 = make_condition_dict(num_nets=3,mcts_d=[3],n_bnds=[8,35],seed=2,repeats=25,steps=25)
cond5 = make_condition_dict(num_nets=2,mcts_d=[4,7],n_bnds=[8,35],seed=1,repeats=25,steps=25)

exp_dict = Dict(:name=>"mturk_fast", :conditions=>Dict(1=>cond1, 2=>cond2, 3=> cond3,4=> cond4,5=> cond5),
                :xQ=>Dict(:inpts=>[:tprob,:N],:epocs=>100,:cmp=>[2,3,4,5],:nn_loc=>"nn_logs"))

fldr = "logs"
for x in keys(exp_dict[:conditions])
    fname = make_useful_fname(exp_dict,x)
    exp_dict[:conditions][x][:fname] = fname
    exp_dict[:conditions][x][:fldr] = fldr
end
