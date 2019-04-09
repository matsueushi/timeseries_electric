# GaussianRandomWalk for Mamba


# Reference:
# Create User-Defined Multivariate Distribution
# https://mambajl.readthedocs.io/en/latest/mcmc/distributions.html#user-defined-univariate-distributions


using Distributed


@everywhere extensions = quote

    using Distributions
    import Distributions: length, insupport, _logpdf

    #################### Gaussian Random Walk ####################
    # 
    # \begin{align*}
    # Y_0 &= D,\\
    # Y_{i+1} &= Y_i+\mu_i + \epsilon_i,\ \epsilon_i \sim \mbox{Normal}(0, \sigma)\\
    # \end{align*}

    mutable struct GaussianRandomWalk <: ContinuousMultivariateDistribution
        mu::Vector{Float64}
        sig::Float64
        init::ContinuousUnivariateDistribution
    end

    length(d::GaussianRandomWalk) = 1 + length(d.mu)

    function insupport(d::GaussianRandomWalk, x::AbstractVector{T}) where {T <: Real}
        length(d) == length(x) && all(isfinite.(x))
    end

    function _logpdf(d::GaussianRandomWalk, x::AbstractVector{T}) where {T <: Real}
        randomwalk_like = logpdf.(Normal.(d.mu + x[1:end - 1], d.sig), x[2:end])
        logpdf(d.init, x[1]) + sum(randomwalk_like)
    end


    #################### SeasonalRandomWalk ####################
    # 
    # S_t &= - \sum_j^{11}S_{t-j} + \omega_t, \\
    # \omega_t &\sim \mbox{Normal}(0, \sigma_\omega^2), \\

    mutable struct SeasonalRandomWalk <: ContinuousMultivariateDistribution
        mu::Vector{Float64}
        sig::Float64
        init::ContinuousMultivariateDistribution
    end

    length(d::SeasonalRandomWalk) = length(d.init) + length(d.mu)

    function insupport(d::SeasonalRandomWalk, x::AbstractVector{T}) where {T <: Real}
        length(d) == length(x) && all(isfinite.(x))
    end

    function _logpdf(d::SeasonalRandomWalk, x::AbstractVector{T}) where {T <: Real}
        init_length = length(d.init)
        x1 = [-sum(view(x, i:i + init_length - 1)) for i = 1:length(x) - init_length]
        randomwalk_like = logpdf.(Normal.(d.mu + x1, d.sig), x[init_length + 1:end])
        logpdf(d.init, x[1:init_length]) + sum(randomwalk_like)
    end

end

@everywhere eval(extensions)