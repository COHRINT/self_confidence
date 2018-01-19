using JLD

include("network_library.jl")

srand(1) # make this reproducible
training_set_size = Int(20e3)

seed_list = collect(1:training_set_size)

base_reward = 2000.
exit_rwd_range = 1000.
exit_rwd_rng = base_reward + (rand(training_set_size)*exit_rwd_range)-exit_rwd_range/2

sensor_rwd_range = 100
sensor_rwd_rng = 0.0 - rand(training_set_size)*sensor_rwd_range

caught_rwd_range = exit_rwd_range
caught_rwd_rng = base_reward + (rand(training_set_size)*caught_rwd_range)-caught_rwd_range/2

degree_rng = rand(collect(3.0:7.0),training_set_size)

n_rng = rand(collect(15:75),training_set_size)

fname = "training_nets.jld"
@save "$fname" training_set_size seed_list base_reward exit_rwd_rng exit_rwd_range sensor_rwd_range sensor_rwd_rng caught_rwd_rng caught_rwd_range degree_rng n_rng

for i=1:training_set_size
    try
        println("Making network $i of $training_set_size")
        g = rand_network(n_rng[i],exit_rwd=exit_rwd_rng[i],caught_rwd=caught_rwd_rng[i],sensor_rwd=sensor_rwd_rng[i],net_seed=seed_list[i],target_mean_degree=degree_rng[i])
        jldopen("$fname", "r+") do file
            file["g$i"] = g
        end
    catch
        println("Failed making network $i, moving on...")
        continue
    end
end

