if nworkers() < 5 && nworkers() > 1
    nw = 150
    println("Adding $nw workers...")
    addprocs(nw, topology=:master_slave)
    println("done")
end

println("including libraries")
@everywhere using Roadnet_MDP
@everywhere importall POMDPs, POMDPToolbox
@everywhere include("make_training_nets.jl")
@everywhere include("make_training_data_SQ.jl")
@everywhere include("send_mail.jl")
include("utilities.jl")
println("done")

experiment_name = "mturk_supplement"
include("experiment_utilities.jl")

create_nets= true
if create_nets
    for cond in keys(exp_dict[:conditions])
        cond_dict = exp_dict[:conditions][cond]
        info("making nets for $(cond_dict[:fname])")
        make_nets(cond_dict)
    end
end

println("creating training data")
make_simulations = true
if make_simulations
    for cond in keys(exp_dict[:conditions])
        cond_dict = exp_dict[:conditions][cond]
        info("running sims for $(cond_dict[:fname])")
        make_training_data(cond_dict)
    end
end

if on_gcloud()
    #if we're on Google cloud, then send an email when code is done
    hname = on_gcloud(return_name=true)
    body = "The code on $hname has finished running"
    subject = "Done running code on $hname"
    send_mail(subject,body)
end
