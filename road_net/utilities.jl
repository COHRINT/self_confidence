function latin_hypercube_sampling{T}(mins::AbstractVector{T}, maxs::AbstractVector{T}, numSamples::Integer)
    # from BlackBoxOptim package
    dims = length(mins)
    result = zeros(T, numSamples, dims)
    @inbounds for i in 1:dims
        interval_len = (maxs[i] - mins[i]) / numSamples
        result[:,i] = shuffle!(linspace(mins[i], maxs[i] - interval_len, numSamples) +
                               interval_len*rand(numSamples))
    end
    return result'
end

function return_indices(ary::Vector)
    u = unique(ary)
    idx = enumerate(u)
    ticks = similar(u)
    labels = Array{String}(size(u))

    indices = Vector{Int64}(length(ary))
    for x in idx
        indices[ary.==x[2]] = x[1]
        ticks[x[1]] = x[1]
        labels[x[1]] = "$(x[2])"
    end

    rng = (minimum(u),maximum(u))

    return indices, ticks, labels, rng
end


## Code below is not fully implemented

function my_lhs(n::Int64,p::Int64;smooth::Symbol=:off,iterations::Int64=5,criterion::Symbol=:maxmin)
    @assert n > 0
    @assert p > 0
    @assert smooth in [:off,:on]
    @assert criterion in [:maxmin,:correlation,:none]

    if crit == :correlation
        error("The '$crit' function has not been implemented yet")
    elseif crit == :maximin
        error("The '$crit' function has not been implemented yet")
    else
        error("argument not recognized")
    end

end

function getsample(n::Int64,p::Int64,smooth::Symbol)::Array{Float64}
    x = rand(n,p)

    for i = 1:p
        x(:,i) = rank(x(:,i))
    end
    if smooth == :on
        x = x - rand(size(x))
    else
        x = x - 0.5
    end
    return x/n
end

function score(x::Int64,crit::Symbol)
    if size(x,1) < 2
        # score is meaningless with just one point
        return 0.
    end
    if crit == :correlation
        error("The '$crit' function has not been implemented yet")
    elseif crit == :maximin
        error("The '$crit' function has not been implemented yet")
    else
        error("argument not recognized")
    end
end

function lhs_rank(x::Vector{Float64})::Vector{Float64}
    z = sort(x)
    r = similar(x)
    for i = 1:length(z)
        r[z[i]] = i
    end
    return r
end

function load_network(nn_prefix::String, epoch::Int,sq_fname::String)
    # the default one doesn't work for some reason, so I'll do it by hand
    println(nn_prefix)
    arch1, arg_params1, aux_params1 = mx.load_checkpoint(string(nn_prefix,"_net1"),epoch)
    arch2, arg_params2, aux_params2 = mx.load_checkpoint(string(nn_prefix,"_net2"),epoch)

    net1 = mx.FeedForward(arch1)
    net1.arg_params = arg_params1
    net1.aux_params = aux_params1

    net2 = mx.FeedForward(arch2)
    net2.arg_params = arg_params2
    net2.aux_params = aux_params2

    sq_data = JLD.load(sq_fname)
    input_sch = sq_data["input_sch"]
    output_sch = sq_data["output_sch"]
    range = sq_data["range"]

    SQ = SQ_model(input_sch,output_sch,net1,net2,range)
    return SQ
end
function make_label_from_keys(d::Dict)
    z = ""
    for x in collect(keys(d))
        z = string(z,x)
    end
    return z
end
function searchdir(path,key1)
    # list of files matching key1
    filt_list = filter(x->contains(x,key1),readdir(path))

    return filt_list
end
function searchdir(path,key1,key2)
    # list of files matching key1
    filt_list = searchdir(path,key1)

    # subset of that list that contains key2
    filt_list2 = filter(x->contains(x,key2),filt_list)

    return filt_list2
end
