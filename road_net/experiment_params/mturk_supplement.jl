# vary transition prob, mcts_depth, and number of nodes in network
cond1 = make_condition_dict(num_nets=500,mcts_d=[10],n_bnds=[8,35],seed=5,role=:trusted)
cond4 = make_condition_dict(num_nets=500,mcts_d=[3],n_bnds=[8,35],seed=2)

exp_dict = Dict(:name=>"mturk_supplement", :conditions=>Dict(1=>cond1, 4=> cond4),
                :xQ=>Dict(:inpts=>[:tprob,:N],:epocs=>1000,:cmp=>[4],:nn_loc=>"nn_logs"))

fldr = "logs"
for x in keys(exp_dict[:conditions])
    fname = make_useful_fname(exp_dict,x)
    exp_dict[:conditions][x][:fname] = fname
    exp_dict[:conditions][x][:fldr] = fldr
end
