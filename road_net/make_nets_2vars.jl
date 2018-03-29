@everywhere using Roadnet_MDP
@everywhere importall POMDPs, POMDPToolbox
@everywhere include("make_training_nets.jl")
@everywhere include("make_training_data_SQ.jl")

fname = "transition_vary"
train_txt = "_reference_solver_training"
test_txt = "_reference_solver_test"
bad_txt = "_bad_solver"
ok_txt = "_ok_solver"

if fname == "transition_vary"
    num_train_nets = 750
    train_t_bnds = [0.0,1.0]
    train_e_bnds = [1000.]
    train_depth = [8]

    num_ok_nets = 250
    ok_t_bnds = [0.0,1.0]
    ok_e_bnds = [1000.]
    ok_depth = [3]

    num_bad_nets = 250
    bad_t_bnds = [0.0,1.0]
    bad_e_bnds = [1000.]
    bad_depth = [1]
elseif fname == "transition_e_vary"
    num_train_nets = 750
    train_t_bnds = [0.0,1.0]
    train_e_bnds = [990.,1000.]
    train_depth = [8]

    num_ok_nets = 250
    ok_t_bnds = [0.0,1.0]
    ok_e_bnds = [10.,500.]
    ok_depth = [3]

    num_bad_nets = 250
    bad_t_bnds = [0.0,1.0]
    bad_e_bnds = [10.,100.]
    bad_depth = [1]
end

create_nets= false
if create_nets
    println("making training solver networks")
    make_nets(num_train_nets,fname="logs/$(fname)$(train_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=train_depth,mcts_e_bounds=train_e_bnds,trans_prob_bounds=train_t_bnds,discount_fact_bounds=[0.95],net_type=:original,random_seed=12345)

    #  println("making test solver networks")
    #  make_nets(250,fname="logs/$(fname)$(test_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[8],mcts_e_bounds=[10.,1000.],trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=2345)

    #  println("making test bad solver networks")
    #  make_nets(num_bad_nets,fname="logs/$(fname)$(bad_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=bad_depth,mcts_e_bounds=bad_e_bnds,trans_prob_bounds=bad_t_bnds,discount_fact_bounds=[0.95],net_type=:original,random_seed=345)
#
    #  println("making test ok solver networks")
    #  make_nets(num_ok_nets,fname="logs/$(fname)$(ok_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=ok_depth,mcts_e_bounds=ok_e_bnds,trans_prob_bounds=ok_t_bnds,discount_fact_bounds=[0.95],net_type=:original,random_seed=45)

    #  println("making mixed solver networks")
    #  make_nets(750,fname="logs/$(fname)$(mixed_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[1,10],mcts_e_bounds=[10.,1300.],trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=95)
end

println("creating training data")
make_simulations = true
if make_simulations

    make_training_data(data_fname="logs/$(fname)$(train_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    #  make_training_data(data_fname="logs/$(fname)$(test_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    make_training_data(data_fname="logs/$(fname)$(bad_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    make_training_data(data_fname="logs/$(fname)$(ok_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    #  make_training_data(data_fname="logs/$(fname)$(mixed_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
end
