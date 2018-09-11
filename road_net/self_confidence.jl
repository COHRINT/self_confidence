using Base.Test
using Distributions

function UPM_LPM(rwd::Vector{Float64};threshold::Float64=0.)::Float64
    # Upper Partial Moment/ Lower Partial Moment
    # Discrete implementation based on a vector of reward estimates

    num_data = length(rwd)
    rwd_prob = similar(rwd)
    for i in unique(rwd)
        num_i = length(rwd[rwd.==i])
        rwd_prob[rwd.==i] = 1./num_data
    end

    # integrate UPM and LPM, using `threshold` as the lower/upper limit respectively
    upm_data = rwd.>=threshold
    lpm_data = rwd.<threshold

    #  println("thresh: $threshold")
    upm = sum(abs.(rwd[upm_data]) .* rwd_prob[upm_data])
    lpm = sum(abs.(rwd[lpm_data]) .* rwd_prob[lpm_data])
    #  println(upm)
    #  println(lpm)

    upm_lpm = upm/lpm
end

function general_logistic(x::Float64;k::Float64=1.,x0::Float64=0.,L::Float64=2.)
    # using logistic function https://en.wikipedia.org/wiki/Logistic_function
    return L/(1+exp(-k*(x-x0)))
end
function general_logistic(x::Array{Float64};k::Float64=1.,x0::Float64=0.,L::Float64=2.)
    return L./(1+exp.(-k.*(x-x0)))
end

function X3(r::Array{Float64})
    return [mean(r) std(r)]
end
function X3(c::Array{Float64},t::Array{Float64};
            global_rwd_range::Array{Float64}=[1.,1.],L::Float64=2.,x0::Float64=0.,
            return_raw_sq::Bool=false,alpha::Float64=1/2)
    sc = std(c)
    st = std(t)
    if sc == 0.
        # if std == 0, replace it with a very small one
        # helps get past some strange simulation conditions where every outcome is identical
        sc = abs(0.001 * mean(c))
        println("std(c) too small replacing with:")
        println("mean:$(mean(c)), std:$(sc)")
    end
    if st == 0.
        st = abs(0.001 * mean(t))
        println("std(t) too small replacing with:")
        println("mean:$(mean(t)), std:$(st)")
    end

    c = Normal(mean(c),sc)
    t = Normal(mean(t),st)
    if return_raw_sq
        (SQ,sq,k) = X3(c,t;global_rwd_range=global_rwd_range,L=L,x0=x0,return_raw_sq=return_raw_sq,alpha=alpha)
        return SQ,sq,k
    else
        SQ = X3(c,t;global_rwd_range=global_rwd_range,L=L,x0=x0,return_raw_sq=return_raw_sq,alpha=alpha)
        return SQ
    end
end
function X3(c::Distributions.Distribution,t::Distributions.Distribution;
            global_rwd_range::Array{Float64}=[1.],L::Float64=2.,x0::Float64=0.,
            return_raw_sq::Bool=false,alpha::Float64=1/2)

    # distance between means, in the global scale
    if length(global_rwd_range) == 1
        # providing range fraction directly
        f = global_rwd_range[1]
    elseif length(global_rwd_range) == 2
        f = abs(mean(c)-mean(t))/(global_rwd_range[2]-global_rwd_range[1])
    else
        error("global_rwd_range length $(length(global_rwd_range)) is not supported")
    end

    # amount of overlap -- 0: identical, 1: no overlap
    # no sign to indicate `direction' of overlap
    H = hellinger_normal(c,t)
    sgn = sign(mean(c)-mean(t))

    H_scaled = sgn*(f^alpha)*H

    k = 5.

    SQ = general_logistic(H_scaled,k=k,x0=x0,L=L)

    println("###############")
    println("mu_c/s_c: $(mean(c))/$(std(c)), mu_t/s_t: $(mean(t))/$(std(t))")
    println("alpha: $alpha")
    println("H: $(H), diff_frac: $f, H_scale: $(H_scaled)")
    println("global rwd: $global_rwd_range")
    println("SQ: $SQ")
    println("###############")

    if return_raw_sq
        D = Dict(:SQ=>SQ,:c=>c,:t=>t,:f=>f,:alpha=>alpha,:H=>H,:k=>k,:x0=>x0,:L=>L,:global_rwd_range=>global_rwd_range)
        #  println(D)
        return SQ, D

        #  return SQ,mean(c),mean(t),f,alpha,D
    else
        return SQ
    end
end
function hellinger_normal(μ_1::Float64,σ_1::Float64,μ_2::Float64,σ_2::Float64)
    c = Normal(μ_1,σ_1)
    t = Normal(μ_2,σ_2)
    return hellinger_normal(c,t)
end
function hellinger_normal(c::Distributions.Distribution,t::Distributions.Distribution)
    # formula from wikipedia
    # https://en.wikipedia.org/wiki/Hellinger_distance
    μ_c = mean(c)
    σ²_c = var(c)
    σ_c = std(c)


    μ_t = mean(t)
    σ²_t = var(t)
    σ_t = std(t)

    H² = 1-sqrt(2*σ_c*σ_t/(σ²_c+σ²_t))*exp(-0.25*(μ_c-μ_t)^2/(σ²_c+σ²_t))
    H = sqrt(H²)

    return H

end

function bhattacharyya_normal(μ_1::Float64,σ_1::Float64,μ_2::Float64,σ_2::Float64)
    c = Normal(μ_1,σ_1)
    t = Normal(μ_2,σ_2)
    return bhattacharyya_normal(c,t)
end
function bhattacharyya_normal(c::Distributions.Distribution,t::Distributions.Distribution)
    μ_c = mean(c)
    σ²_c = var(c)


    μ_t = mean(t)
    σ²_t = var(t)

    D = 1/4*log(1/4*(σ²_c/σ²_t + σ²_t/σ²_c + 2)) + 1/4*((μ_c-μ_t)^2/(σ²_c/σ²_t))

end

function X4(rwd::Vector{Float64};threshold::Float64=0.,k::Float64=1.)
    # OUTCOME ASSESSMENT
    # rwd is a raw vector of rewards from separate simulations
    # calculate the probability of each unique value
    upm_lpm = UPM_LPM(rwd,threshold=threshold)

    outcome_assessment = 2./(1.+exp(-k*(log(upm_lpm))))-1.

end

# tests from Matt's thesis, Section 5.3
@test X4([10.]) ≈ 1.
@test X4([100.]) ≈ 1.
@test X4([-10.]) ≈ -1.
@test X4([-100.]) ≈ -1.

@test X4([-5.,25.]) ≈ 2/3
@test X4([-25.,5.]) ≈ -2/3

@test X4([-10.,10.]) ≈ 0.
@test X4([-100.,100.]) ≈ 0.

@test X4([-10.,10.,10.,10.]) ≈ 1/2
@test X4([10.,-10.,-10.,-10.]) ≈ -1/2
@test X4([-100.,100.,100.,100.]) ≈ 1/2
@test X4([100.,-100.,-100.,-100.]) ≈ -1/2

# tests for X3
#  @test X3(Normal(0.,1.),Normal(0.,1.)) ≈ 1.
#  @test X3(Normal(100.,2.),Normal(0.,1.)) ≈ 2.
#  @test X3(Normal(100.,2.),Normal(0.,1.),L=3.) ≈ 3.
#  @test X3(Normal(-10e9,2.),Normal(0.,1.)) ≈ 0.
#  @test X3(Normal(100.,2.),Normal(0.,1.),global_rwd_range=(-1e12,1e12)) ≈ 1.
#  @test X3(Normal(100.,2.),Normal(0.,1.),global_rwd_range=(-1e-12,1e-12)) ≈ 2.
#  @test X3(Normal(-1000,2.),Normal(0.,1.),global_rwd_range=(-1.,1.)) ≈ 0.
@test general_logistic(0.,L=1.,k=1.,x0=0.) ≈ 0.5
@test general_logistic(1.,L=2.,k=100.,x0=0.) ≈ 2.
@test general_logistic(-10.,L=2.,k=100.,x0=0.) ≈ 0.
@test general_logistic(5.,L=2.,k=100.,x0=5.) ≈ 1.
