cond1 = make_condition_dict(num_nets=750,solver_depth=[8],net_type=:original,seed=5,
                            e_bnds=[10.,1000.],role=:trusted)
cond2 = make_condition_dict(num_nets=250,solver_depth=[3],net_type=:original,seed=4,
                            e_bnds=[10.,1000.],role=:candidate)
cond3 = make_condition_dict(num_nets=250,solver_depth=[1],net_type=:original,seed=3,
                            e_bnds=[10.,1000.],role=:candidate)

exp_dict = Dict(:name=>"transition_e_vary", :conditions=>Dict(1=>cond1, 2=>cond2, 3=> cond3))

fldr = "logs"
for x in keys(exp_dict[:conditions])
    fname = make_useful_fname(exp_dict,x)
    exp_dict[:conditions][x][:fname] = fname
    exp_dict[:conditions][x][:fldr] = fldr
end
