include("make_training_nets.jl")
include("make_training_data_SQ.jl")

fname = "transition_e_vary"
train_txt = "_reference_solver_training"
test_txt = "_reference_solver_test"
bad_txt = "_bad_solver"
ok_txt = "_ok_solver"
mixed_txt = "_mixed_solver"

println("making training solver networks")
make_nets(750,fname="logs/$(fname)$(train_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[8],mcts_e_bounds=[1000.],
          trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=12345)
println("making test solver networks")
make_nets(250,fname="logs/$(fname)$(test_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[8],mcts_e_bounds=[1000.],
          trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=2345)
println("making test bad solver networks")
make_nets(250,fname="logs/$(fname)$(bad_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[1],mcts_e_bounds=[0.,1000.],
          trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=345)
println("making test ok solver networks")
make_nets(250,fname="logs/$(fname)$(ok_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[3],mcts_e_bounds=[0.,1000.],
          trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=45)
println("making mixed solver networks")
make_nets(750,fname="logs/$(fname)$(ok_txt).jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[0,10],mcts_e_bounds=[0.,1300.],
          trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=5)

println("creating training data")

make_training_data(data_fname="logs/$(fname)$(train_txt)")
make_training_data(data_fname="logs/$(fname)$(test_txt)")
make_training_data(data_fname="logs/$(fname)$(bad_txt)")
make_training_data(data_fname="logs/$(fname)$(ok_txt)")
make_training_data(data_fname="logs/$(fname)$(mixed_txt)")

