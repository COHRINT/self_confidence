using Distributions
using Plots, StatPlots

function sigmoid(x::Float64;scale::Float64=1.,offset::Float64=0.)
    if isinf(exp(x))
        return scale
    else
        return scale.*exp(x)/(exp(x)+1.) - offset
    end
end

function s1(c::Distributions.Distribution,t::Distributions.Distribution;
                                      solver_rwd_range::Float64=1.)
    t1 = c.μ/c.σ
    t2 = t.μ/t.σ
    t3 = t1 - t2
    t4 = t3/solver_rwd_range
    #  println("c_mu: $(c.μ), c_std: $(c.σ)")
    println("t1: $t1, t2: $t2, t3: $t3, t4: $t4")
    return sigmoid(t4,scale=2.,offset=0.)
end
function s2(c::Distributions.Distribution,t::Distributions.Distribution;
                                      solver_rwd_range::Float64=1.)
    t1 = c.μ/c.σ
    t2 = t.μ/t.σ
    t3 = t1 - t2
    t4 = t3/solver_rwd_range
    #  println("c_mu: $(c.μ), c_std: $(c.σ)")
    println("t1: $t1, t2: $t2, t3: $t3, t4: $t4")
    val = NaN
    if c.μ == 0. && t.μ == 0.
        println("HERE")
        val = sigmoid((1/c.σ - 1/t.σ)/solver_rwd_range,scale=2.,offset=0.)
    else
        val = sigmoid(t4,scale=2.,offset=0.)
    end
    return val
end

t_mean = zeros(5)
t_std = ones(5)
c_mean = zeros(5)
c_std = collect(0.7:1:5)

t_dist = []
c_dist = []
sq = []
sq2 = []

for i = 1:length(t_mean)
    push!(t_dist,Normal(t_mean[i],t_std[i]))
    push!(c_dist,Normal(c_mean[i],c_std[i]))
    push!(sq,s1(c_dist[end],t_dist[end]))
    push!(sq2,s2(c_dist[end],t_dist[end]))
end

plot(c_std,sq)
plot!(c_std,sq2)
