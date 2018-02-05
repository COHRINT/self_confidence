using Base.Test

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

    upm = sum(rwd[upm_data] .* rwd_prob[upm_data])
    lpm = sum(abs.(rwd[lpm_data]) .* rwd_prob[lpm_data])

    upm_lpm = upm/lpm
end

function X3(rwd::Vector{Float64};train::Bool=false)::Array{Float64}
    # SOLVER QUALITY
    # THIS ACTUALLY NEEDS TO ACCESS SOME MODEL, AND THEN PREDICT BASED ON PROBLEM AND SOVLER PARAMETERS
    if train
        # return data for the training set
        thresh = median(rwd)
        upm_lpm = UPM_LPM(rwd,threshold=thresh)
        return [thresh upm_lpm]
    else
        # return predictions
        error("X3 test not implemented yet")

        predicted_threshold = 0.
        predicted_upm_lpm = 0.

        return [predicted_threshold predicted_upm_lpm]
    end
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
