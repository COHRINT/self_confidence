@everywhere using Roadnet_MDP
@everywhere importall POMDPs, POMDPToolbox
@everywhere include("make_training_nets.jl")
@everywhere include("make_training_data_SQ.jl")
@everywhere include("send_mail.jl")

fname = "n_vary"
train_txt = "_reference_solver_training"
test_txt = "_reference_solver_test"
bad_txt = "_bad_solver"
ok_txt = "_ok_solver"

if fname == "transition_vary"
    include("experiment_params/transition_vary.jl")
elseif fname == "n_vary"
    include("experiment_params/n_vary.jl")
elseif fname == "sense_vary"
    include("experiment_params/sense_vary.jl")
elseif fname == "transition_e_vary"
    include("experiment_params/transition_e_vary.jl")
elseif fname == "net_transition_vary"
    include("experiment_params/net_transition_vary.jl")
elseif fname == "net_transition_sense_vary"
    include("experiment_params/net_transition_sense_vary.jl")
elseif fname == "something else"
    #keeping empty for now
end

create_nets= true
if create_nets
    println("making training solver networks")
    make_nets(num_train_nets,fname="logs/$(fname)$(train_txt).jld",exit_rwd_bounds=training_exit_rwd_bnds,sensor_rwd_bounds=training_sense_rwd_bnds,caught_rwd_bounds=training_caught_rwd_bnds,degree_bounds=train_deg_bnds,n_bounds=train_n_bnds,mcts_its_bounds=training_mcts_bnds,mcts_depth_bounds=train_depth,mcts_e_bounds=train_e_bnds,trans_prob_bounds=train_t_bnds,discount_fact_bounds=train_d_bnds,net_type=train_net_type,random_seed=12345)

    println("making test bad solver networks")
    make_nets(num_bad_nets,fname="logs/$(fname)$(bad_txt).jld",exit_rwd_bounds=bad_exit_rwd_bnds,sensor_rwd_bounds=bad_sense_rwd_bnds,caught_rwd_bounds=bad_caught_rwd_bnds,degree_bounds=bad_deg_bnds,n_bounds=bad_n_bnds,mcts_its_bounds=bad_mcts_bnds,mcts_depth_bounds=bad_depth,mcts_e_bounds=bad_e_bnds,trans_prob_bounds=bad_t_bnds,discount_fact_bounds=bad_d_bnds,net_type=bad_net_type,random_seed=345)
#
    println("making test ok solver networks")
    make_nets(num_ok_nets,fname="logs/$(fname)$(ok_txt).jld",exit_rwd_bounds=ok_exit_rwd_bnds,sensor_rwd_bounds=ok_sense_rwd_bnds,caught_rwd_bounds=ok_caught_rwd_bnds,degree_bounds=ok_deg_bnds,n_bounds=ok_n_bnds,mcts_its_bounds=ok_mcts_bnds,mcts_depth_bounds=ok_depth,mcts_e_bounds=ok_e_bnds,trans_prob_bounds=ok_t_bnds,discount_fact_bounds=bad_d_bnds,net_type=ok_net_type,random_seed=45)
end

println("creating training data")
make_simulations = true
if make_simulations
    make_training_data(data_fname="logs/$(fname)$(train_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    make_training_data(data_fname="logs/$(fname)$(bad_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    make_training_data(data_fname="logs/$(fname)$(ok_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
end

if on_gcloud()
    #if we're on Google cloud, then send an email when code is done
    hname = on_gcloud(return_name=true)
    body = "The code on $hname has finished running"
    subject = "Done running code on $hname"
    send_mail(subject,body)
end
