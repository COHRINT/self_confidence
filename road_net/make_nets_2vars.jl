@everywhere using Roadnet_MDP
@everywhere importall POMDPs, POMDPToolbox
@everywhere include("make_training_nets.jl")
@everywhere include("make_training_data_SQ.jl")

fname = "net_transition_vary"
train_txt = "_reference_solver_training"
test_txt = "_reference_solver_test"
bad_txt = "_bad_solver"
ok_txt = "_ok_solver"

if fname == "transition_vary"
    num_train_nets = 750
    train_t_bnds = [0.0,1.0]
    train_e_bnds = [1000.]
    train_depth = [8]
    train_deg_bnds = [4.]
    train_net_type = :original
    train_n_bnds = [13]

    num_ok_nets = 250
    ok_t_bnds = [0.0,1.0]
    ok_e_bnds = [1000.]
    ok_depth = [3]
    ok_deg_bnds = [4.]
    ok_net_type = :original
    ok_n_bnds = [13]

    num_bad_nets = 250
    bad_t_bnds = [0.0,1.0]
    bad_e_bnds = [1000.]
    bad_depth = [1]
    bad_deg_bnds = [4.]
    bad_net_type = :original
    bad_n_bnds = [13]
elseif fname == "transition_e_vary"
    num_train_nets = 750
    train_t_bnds = [0.0,1.0]
    train_e_bnds = [10.,1000.]
    train_depth = [8]
    train_deg_bnds = [4.]
    train_net_type = :original
    train_n_bnds = [13,45]
    train_n_bnds = [13]

    num_ok_nets = 250
    ok_t_bnds = [0.0,1.0]
    ok_e_bnds = [10.,1000.]
    ok_depth = [3]
    ok_deg_bnds = [4.]
    ok_net_type = :original
    ok_n_bnds = [13]

    num_bad_nets = 250
    bad_t_bnds = [0.0,1.0]
    bad_e_bnds = [10.,1000.]
    bad_depth = [1]
    bad_deg_bnds = [4.]
    bad_net_type = :original
    bad_n_bnds = [13]
elseif fname == "net_transition_vary"
    num_train_nets = 7
    train_t_bnds = [0.0,1.0]
    train_e_bnds = [1000.]
    train_depth = [8]
    train_deg_bnds = [3.,8.]
    train_net_type = :random
    train_n_bnds = [13,45]

    num_ok_nets = 2
    ok_t_bnds = [0.0,1.0]
    ok_e_bnds = [1000.]
    ok_depth = [3]
    ok_deg_bnds = [3.,8.]
    ok_net_type = :random
    ok_n_bnds = [13,45]

    num_bad_nets = 2
    bad_t_bnds = [0.0,1.0]
    bad_e_bnds = [1000.]
    bad_depth = [1]
    bad_deg_bnds = [3.,8.]
    bad_net_type = :random
    bad_n_bnds = [13,45]

elseif fname == "something else"
    num_train_nets = 7
    train_t_bnds = [0.0,1.0]
    train_e_bnds = [1000.]
    train_depth = [8]
    train_deg_bnds = [3.,8.]
    train_net_type = :random
    train_n_bnds = [13,45]

    num_ok_nets = 2
    ok_t_bnds = [0.0,1.0]
    ok_e_bnds = [1000.]
    ok_depth = [3]
    ok_deg_bnds = [3.,8.]
    ok_net_type = :random
    ok_n_bnds = [13,45]

    num_bad_nets = 2
    bad_t_bnds = [0.0,1.0]
    bad_e_bnds = [1000.]
    bad_depth = [1]
    bad_deg_bnds = [3.,8.]
    bad_net_type = :random
    bad_n_bnds = [13,45]
end

create_nets= true
if create_nets
    println("making training solver networks")
    make_nets(num_train_nets,fname="logs/$(fname)$(train_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=train_deg_bnds,n_bounds=train_n_bnds,mcts_its_bounds=[500],mcts_depth_bounds=train_depth,mcts_e_bounds=train_e_bnds,trans_prob_bounds=train_t_bnds,discount_fact_bounds=[0.95],net_type=train_net_type,random_seed=12345)

    println("making test bad solver networks")
    make_nets(num_bad_nets,fname="logs/$(fname)$(bad_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=bad_deg_bnds,n_bounds=bad_n_bnds,mcts_its_bounds=[500],mcts_depth_bounds=bad_depth,mcts_e_bounds=bad_e_bnds,trans_prob_bounds=bad_t_bnds,discount_fact_bounds=[0.95],net_type=bad_net_type,random_seed=345)
#
    println("making test ok solver networks")
    make_nets(num_ok_nets,fname="logs/$(fname)$(ok_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],degree_bounds=ok_deg_bnds,n_bounds=ok_n_bnds,mcts_its_bounds=[500],mcts_depth_bounds=ok_depth,mcts_e_bounds=ok_e_bnds,trans_prob_bounds=ok_t_bnds,discount_fact_bounds=[0.95],net_type=ok_net_type,random_seed=45)
end

println("creating training data")
make_simulations = true
if make_simulations
    make_training_data(data_fname="logs/$(fname)$(train_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    make_training_data(data_fname="logs/$(fname)$(bad_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
    make_training_data(data_fname="logs/$(fname)$(ok_txt)",repeats=250,sim_steps=-1,dis_rwd=false)
end
