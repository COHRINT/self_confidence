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
