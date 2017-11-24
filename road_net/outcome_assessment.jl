using Base.Test
function OA(rwd::Vector{Float64};threshold::Float64=0.,k::Float64=1.)::Float64
    # rwd is a raw vector of rewards from separate simulations
    # calculate the probability of each unique value
    num_data = length(rwd)
    rwd_prob = similar(rwd)
    for i in unique(rwd)
        num_i = length(rwd[rwd.==i])
        rwd_prob[rwd.==i] = 1./num_data
    end

    # integrate UPM and LPM, using `threshold` as the lower/upper limit respectively
    upm_data = rwd.>=threshold
    lpm_data = rwd.<threshold

    upm = sum(rwd[upm_data] .* rwd_prob[upm_data])
    lpm = sum(abs.(rwd[lpm_data]) .* rwd_prob[lpm_data])

    upm_lpm = upm/lpm

    outcome_assessment = 2./(1.+exp(-k*(log(upm_lpm))))-1.

end

# tests from Matt's thesis, Section 5.3
@test OA([10.]) ≈ 1.
@test OA([100.]) ≈ 1.
@test OA([-10.]) ≈ -1.
@test OA([-100.]) ≈ -1.

@test OA([-5.,25.]) ≈ 2/3
@test OA([-25.,5.]) ≈ -2/3

@test OA([-10.,10.]) ≈ 0.
@test OA([-100.,100.]) ≈ 0.

@test OA([-10.,10.,10.,10.]) ≈ 1/2
@test OA([10.,-10.,-10.,-10.]) ≈ -1/2
@test OA([-100.,100.,100.,100.]) ≈ 1/2
@test OA([100.,-100.,-100.,-100.]) ≈ -1/2
