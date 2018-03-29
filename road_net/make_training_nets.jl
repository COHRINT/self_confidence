using JLD

include("network_library.jl")

function n_rand_in_range(A::Array,N::Int64)
    # generate an array of length N within the range specified by A
    # length 1 means parameter is fixed, length 2 means random within bounds
    @assert length(A) == 1 || length(A) == 2
    eltype = typeof(A).parameters[1]
    @assert eltype == Float64 || eltype == Int64

    if length(A) == 1
        # length 1 indicates no range, meaning all are identical
        ary = ones(N)*A[1]

        if eltype == Int64
            return convert(Vector{Int64},ary)
        else
            return ary
        end
    else
       # length 2 means we have the bounds
       if eltype == Int64
           return A[1] + rand(A[1]:A[2],N)
       else
           A_range = diff(A)[1]
           return A[1] + rand(N)*A_range
       end
   end
end

function make_nets(training_set_size::Int64;exit_rwd_bounds::Array{Float64}=[2000.],
                   sensor_rwd_bounds::Array{Float64}=[-100.],caught_rwd_bounds::Array{Float64}=[-2000.],
                   degree_bounds::Array{Float64}=[3.0,7.0],n_bounds::Array{Int64}=[15,75],
                   mcts_its_bounds::Array{Int64}=[100],mcts_depth_bounds::Array{Int64}=[5],
                   mcts_e_bounds::Array{Float64}=[1.],trans_prob_bounds::Array{Float64}=[1.],
                   discount_fact_bounds::Array{Float64}=[0.9],net_type::Symbol=:original,
                   random_seed::Int64=1,fname::String="logs/generated_training_nets.jld")

    srand(random_seed)
    seed_list = collect(1:training_set_size)

    # POMDP Problem Properties
    exit_rwd_ary = n_rand_in_range(exit_rwd_bounds,training_set_size)
    sensor_rwd_ary = n_rand_in_range(sensor_rwd_bounds,training_set_size)
    caught_rwd_ary = n_rand_in_range(caught_rwd_bounds,training_set_size)
    transition_prob_ary = n_rand_in_range(trans_prob_bounds,training_set_size)
    discount_fact_ary = n_rand_in_range(discount_fact_bounds,training_set_size)

    # Solver Properties
    mcts_its_ary = n_rand_in_range(mcts_its_bounds,training_set_size)
    mcts_depth_ary = n_rand_in_range(mcts_depth_bounds,training_set_size)
    mcts_e_ary = n_rand_in_range(mcts_e_bounds,training_set_size)

    # Network Properties
    degree_ary = n_rand_in_range(degree_bounds,training_set_size)
    n_ary = n_rand_in_range(n_bounds,training_set_size)


    @save "$fname" training_set_size seed_list exit_rwd_bounds exit_rwd_ary sensor_rwd_bounds sensor_rwd_ary caught_rwd_bounds caught_rwd_ary degree_bounds degree_ary n_bounds n_ary

    println(exit_rwd_ary,sensor_rwd_ary,caught_rwd_ary,degree_ary,n_ary)

    # store problems in a dictionary
    problem_dict = Dict()

    for i=1:training_set_size
        try
            println("Making network $i of $training_set_size")

            if net_type == :random
                g = rand_network(n_ary[i],exit_rwd=exit_rwd_ary[i],caught_rwd=caught_rwd_ary[i],
                                 sensor_rwd=sensor_rwd_ary[i],net_seed=seed_list[i],
                                 exit_nodes=[8],target_mean_degree=degree_ary[i])
            elseif net_type == :original
                g = original_roadnet(exit_rwd=exit_rwd_ary[i],caught_rwd=caught_rwd_ary[i],sensor_rwd=sensor_rwd_ary[i])
            else
                println("no other net_types implemented at this time")
            end

            evader_start = 1
            pursuer_start = 4
            exit_loc = 13
            display_network(g,evader_locs=[evader_start],pursuer_locs=[pursuer_start],fname="logs/net$i")

            problem_dict[i] = Dict(:graph=>g,:mcts_its=>mcts_its_ary[i],:mcts_depth=>mcts_depth_ary[i],
                                   :mcts_e=>mcts_e_ary[i],:net_seed=>seed_list[i],:n_param=>n_ary[i],
                                   :exit_rwd=>exit_rwd_ary[i],:caught_rwd=>caught_rwd_ary[i],
                                   :discount=>discount_fact_ary[i],:trans_prob=>transition_prob_ary[i],
                                   :sensor_rwd=>sensor_rwd_ary[i],:target_degree=>degree_ary[i],
                                   :evader_start=>evader_start,:pursuer_start=>pursuer_start,:exit_loc=>exit_loc)
        catch
            println("Failed making network $i, moving on...")
            problem_dict[i] = Dict(:error=>"failed making network")
            continue
        end
    end

    println("writing data to $fname")
    jldopen("$fname", "w") do file
        addrequire(file, MetaGraphs)
        write(file,"problem_dict", problem_dict)
    end
end

## run on original road net, varying only transition probability
#  println("making transition networks")
#  make_nets(2500,fname="logs/transition_vary_4.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[8],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original)
#
#  println("making test multi-graph networks")
#  make_nets(250,fname="logs/transition_vary_test_4.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[8],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=12345)

#  println("making test bad solver networks")
#  make_nets(250,fname="logs/transition_vary_test_bad_solver.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[1],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=12345)
#  println("making test ok solver networks")
#  make_nets(250,fname="logs/transition_vary_test_ok_solver.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[500],mcts_depth_bounds=[3],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.0,1.0],discount_fact_bounds=[0.95],net_type=:original,random_seed=12345)

#########################################################
###########Discount Networks
#########################################################
##  run on original road net, varying only discount factor
#  println("making discount networks")
#  make_nets(500,fname="logs/discount_vary.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[5],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.9],discount_fact_bounds=[0.00,1.0],net_type=:original)
#
#  println("making test discount networks")
#  make_nets(50,fname="logs/discount_vary_test.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[5],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.9],discount_fact_bounds=[0.00,1.0],net_type=:original)

#########################################################
###########Random Networks
#########################################################
## run with fixed parameters on changing network
#  println("making random networks")
#  make_nets(500,fname="logs/net_vary.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[5],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.7],discount_fact_bounds=[0.95],net_type=:random)
#
#  println("making test random networks")
#  make_nets(50,fname="logs/net_vary_test.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[5],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.7],discount_fact_bounds=[0.95],net_type=:random)
#
#  println("making test random networks bad solver")
#  make_nets(50,fname="logs/net_vary_test_bad_solver.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[1],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.7],discount_fact_bounds=[0.95],net_type=:random)
#
#  println("making test random networks OK solver")
#  make_nets(50,fname="logs/net_vary_test_bad_solver.jld",exit_rwd_bounds=[2000.],sensor_rwd_bounds=[-100.],caught_rwd_bounds=[-2000.],
          #  degree_bounds=[4.],n_bounds=[13],mcts_its_bounds=[100],mcts_depth_bounds=[3],mcts_e_bounds=[1000.],
          #  trans_prob_bounds=[0.7],discount_fact_bounds=[0.95],net_type=:random)
