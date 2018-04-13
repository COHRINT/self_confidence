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

function populate_net_dict(i,training_set_size,n,exit_rwd,caught_rwd,sensor_rwd,seed_list,degree,mcts_its,mcts_depth,mcts_e,discount_fact,transition_prob,fname,net_type)
    problem_dict = Dict()
        try
            println("Making network $i of $training_set_size")

            if net_type == :random
                g = rand_network(n,exit_rwd=exit_rwd,caught_rwd=caught_rwd,
                                 sensor_rwd=sensor_rwd,net_seed=seed_list,approx_E=2*n,
                                 exit_nodes=[8],target_mean_degree=degree,method=:erdos_n_e)
            elseif net_type == :original
                g = original_roadnet(exit_rwd=exit_rwd,caught_rwd=caught_rwd,sensor_rwd=sensor_rwd)
            else
                println("no other net_types implemented at this time")
            end

            evader_start = 1
            pursuer_start = 4
            exit_loc = 13
            display_network(g,evader_locs=[evader_start],pursuer_locs=[pursuer_start],fname="logs/net$i")

            problem_dict[i] = Dict(:graph=>g,:mcts_its=>mcts_its,:mcts_depth=>mcts_depth,
                                   :mcts_e=>mcts_e,:net_seed=>seed_list,:n_param=>n,
                                   :exit_rwd=>exit_rwd,:caught_rwd=>caught_rwd,
                                   :discount=>discount_fact,:trans_prob=>transition_prob,
                                   :sensor_rwd=>sensor_rwd,:target_degree=>degree,
                                   :evader_start=>evader_start,:pursuer_start=>pursuer_start,
                                   :exit_loc=>exit_loc)
        catch
            println("Failed making network $i, moving on...")
            problem_dict[i] = Dict(:error=>"failed making network")
        end
    return problem_dict
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
    pd_array = pmap(populate_net_dict,collect(1:training_set_size),repmat([training_set_size],training_set_size,1),n_ary,exit_rwd_ary,caught_rwd_ary,sensor_rwd_ary,seed_list,degree_ary,mcts_its_ary,mcts_depth_ary,mcts_e_ary,discount_fact_ary,transition_prob_ary,repmat([fname],training_set_size,1),repmat([net_type],training_set_size,1))
    # pmap gives an array, we want a Dict, so put the entries into a dict
    # ther must be a better way to do this, but I don't have time to figure it out
    problem_dict = Dict()
    for i=1:length(pd_array)
        #  display(pd_array[i][i])
        # HACK, HACKETY HACK HACK HACK
        problem_dict[i] = pd_array[i][i]
    end
    #  display(problem_dict)
    #  display(problem_dict[1])
    #  display(problem_dict[2])
    #  error()

    println("writing data to $fname")
    jldopen("$fname", "w") do file
        addrequire(file, MetaGraphs)
        write(file,"problem_dict", problem_dict)
    end
end
