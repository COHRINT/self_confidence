num_train_nets = 750
train_t_bnds = [0.0,1.0]
train_e_bnds = [1000.]
train_depth = [8]
train_deg_bnds = [4.]
train_net_type = :random
train_n_bnds = [8,120]
train_d_bnds = [0.95]
training_exit_rwd_bnds = [2000.]
training_sense_rwd_bnds = [-100.]
training_caught_rwd_bnds = [-2000.]
training_mcts_bnds = [1000]

num_ok_nets = 250
ok_t_bnds = [0.0,1.0]
ok_e_bnds = [1000.]
ok_depth = [3]
ok_deg_bnds = [4.]
ok_net_type = :random
ok_n_bnds = [8,120]
ok_d_bnds = [0.95]
ok_exit_rwd_bnds = [2000.]
ok_sense_rwd_bnds = [-100.]
ok_caught_rwd_bnds = [-2000.]
ok_mcts_bnds = [1000]

num_bad_nets = 250
bad_t_bnds = [0.0,1.0]
bad_e_bnds = [1000.]
bad_depth = [1]
bad_deg_bnds = [4.]
bad_net_type = :random
bad_n_bnds = [8,120]
bad_d_bnds = [0.95]
bad_exit_rwd_bnds = [2000.]
bad_sense_rwd_bnds = [-100.]
bad_caught_rwd_bnds = [-2000.]
bad_mcts_bnds = [1000]
