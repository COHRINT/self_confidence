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
            return_raw_sq::Bool=false)
    c = Normal(mean(c),std(c))
    t = Normal(mean(t),std(t))
    if return_raw_sq
        (SQ,sq,k) = X3(c,t;global_rwd_range=global_rwd_range,L=L,x0=x0,return_raw_sq=return_raw_sq)
        return SQ,sq,k
    else
        SQ = X3(c,t;global_rwd_range=global_rwd_range,L=L,x0=x0,return_raw_sq=return_raw_sq)
        return SQ
    end
end
function X3_old(c::Distributions.Distribution,t::Distributions.Distribution;
            global_rwd_range::Tuple{Float64,Float64}=(-1.,1.),L::Float64=2.,x0::Float64=0.,
            return_raw_sq::Bool=false)
    # using idea of 'standardized moment' https://en.wikipedia.org/wiki/Standardized_moment
    # in this version we are using `glb_rwd_low` as the absolute mean on which both distributions are being compared
    # also we scale the moment a second time by the total range of known solutions for all solvers
    glb_rwd_low = global_rwd_range[1]
    glb_rwd_high = global_rwd_range[2]
    glb_rwd_rng = glb_rwd_high - glb_rwd_low

    # handle degenerate cases where solution can go to Inf
    if isapprox(c.σ,0)
        c_sig = 1.
    else
        c_sig = c.σ
    end
    if isapprox(t.σ,0)
        t_sig = 1.
    else
        t_sig = t.σ
    end

    if c.μ == glb_rwd_low && t.μ == glb_rwd_low
        # degenerate case where c.μ and t.μ are equal (i.e. numerator goes to zero), we still want to compare sigmas
        println("### c.μ == t.μ ###")
        cr = 1/(c_sig)
        tr = 1/(c_sig)
    else
        # standard moments w.r.t. glb_rwd_low
        cr = (c.μ-glb_rwd_low)/(c_sig)
        tr = (t.μ-glb_rwd_low)/(t_sig)
    end

    ref_dist = (glb_rwd_high-glb_rwd_low)/t_sig

    println("cmu: $(c.μ), tmu: $(t.μ), rmu: $glb_rwd_high")
    println("cmu_s: $(c.μ-glb_rwd_low), tmu_s: $(t.μ-glb_rwd_low), rmu: $glb_rwd_rng")
    println("csig: $(c_sig), tsig: $(t_sig), rsig: $(t_sig)")
    println("cr: $cr, tr: $tr, rd: $ref_dist")

    sq = (cr-tr)/ref_dist
    #  sq = (c.μ-t.μ)/abs(c_sig-t_sig)/ref_dist #decided to try something else for a bit
    #  k = 1/5
    k = 5.

    SQ = general_logistic(sq,k=k,x0=x0,L=L)

    println("SQ: $SQ, sq: $sq, k: $k, L: $L, x0: $x0")
    if return_raw_sq
        return SQ,sq,k
    else
        return SQ
    end
end
function X3(c::Distributions.Distribution,t::Distributions.Distribution;
            global_rwd_range::Array{Float64}=[1.],L::Float64=2.,x0::Float64=0.,
            return_raw_sq::Bool=false)

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
    #  D = bhattacharyya_normal(c,t)
    D = hellinger_normal(c,t)
    sgn = sign(mean(c)-mean(t))

    D_scaled = sgn*sqrt(f)*D

    k = 5.

    SQ = general_logistic(D_scaled,k=k,x0=x0,L=L)

    println("###############")
    println("mu_c/s_c: $(mean(c))/$(std(c)), mu_t/s_t: $(mean(t))/$(std(t))")
    println("D: $(D), diff_frac: $f, D_scale: $(D_scaled)")
    println("global rwd: $global_rwd_range")
    println("SQ: $SQ")
    println("###############")

    if return_raw_sq
        return SQ,D_scaled,k
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

    #  return sqrt(H)

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
