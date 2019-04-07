# GaussianRandomWalk for Mamba

# \begin{align*}
# Y_0 &= D,\\
# Y_{i+1} &= Y_i+\mu_i+\epsilon_i,\ \epsilon_i \sim \mbox{Normal}(0, \sigma)\\
# \end{align*}

# Reference:
# Create User-Defined Multivariate Distribution
# https://mambajl.readthedocs.io/en/latest/mcmc/distributions.html#user-defined-univariate-distributions


using Distributed


@everywhere extensions = quote

    using Distributions
    import Distributions: length, insupport, _logpdf

    mutable struct GaussianRandomWalk <: ContinuousMultivariateDistribution
        mu::Vector{Float64}
        sig::Float64
        init::ContinuousUnivariateDistribution
    end

    length(d::GaussianRandomWalk) = length(d.mu) + 1

    function insupport(d::GaussianRandomWalk, x::AbstractVector{T}) where {T <: Real}
        length(d) == length(x) && all(isfinite.(x))
    end

    function _logpdf(d::GaussianRandomWalk, x::AbstractVector{T}) where {T <: Real}
        randomwalk_like = logpdf.(Normal.(d.mu + x[1:end - 1], d.sig), x[2:end])
        logpdf(d.init, x[1]) + sum(randomwalk_like)
    end

end
