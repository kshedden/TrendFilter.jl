
# https://www.tandfonline.com/doi/full/10.1080/10618600.2015.1054033
# https://arxiv.org/abs/1406.2082

mutable struct Trendfilter{T}

    # The x coordinates of the data, must be distinct and increasing
    x::Vector{T}

    # The y coordinates of the data
    y::Vector{T}

    # Optional weight matrix, (y-beta)'*wt*(y-beta) is used to measure goodness as fit.
    # If wt is a vector it is interpreted as a diagonal matrix.
    wt::AbstractArray

    # The current fitted value
    beta::Vector{T}

    # A slack variable
    u::Vector{T}

    # The penalty parameter
    lam::Float64

    # The ADMM weight parameter
    rho::Float64

    # A penalty matrix of one order lower than that used in the loss function.
    Dk::BandedMatrix

    # The penalty matrix, used in the formal optimization problem, not used
    # explicitly by ADMM except to judge convergence.
    Dkp1::BandedMatrix

    # Auxiliary variables
    DktDk::Symmetric
    Dkbeta::Vector{T}
    fl::FusedLasso

    # Various workspaces
    numer::Vector{T}
    denom::Cholesky
    work2::Vector{T}

    # Current iteration number
    iter::Int

    # Indicator that the algorithm converged
    converged::Bool
end

function StatsBase.fit(Trendfilter, x, y, lam; order = 1, dofit = true)
    tf = Trendfilter(x, y, lam; order = order)
    dofit && fit!(tf)
    return tf
end

StatsBase.coef(tf::Trendfilter) = tf.beta

function Trendfilter(x, y, lam; wt = zeros(0, 0), order = 1)
    n = length(y)
    issorted(x) || throw(error("x must be sorted"))
    length(x) == n || throw(error("x and y must have the same length"))
    order in [1, 2] || throw(error("order must be either 1 or 2"))
    n == length(unique(x)) || throw(error("all elements of x must be unique"))

    # Equation 19 of Tibshirani and Ramdas
    rho = lam * (x[end] - x[1])^order / n

    D1, D2, D3 = differencemats(x)
    Dk, Dkp1 = if order == 1
        D1, D2
    elseif order == 2
        D2, D3
    else
        throw(error("invalid order"))
    end
    DktDk = Symmetric(Dk' * Dk)
    Dkbeta = zeros(n - order)
    beta = zeros(n)
    u = zeros(n - order)
    fl = fit(FusedLasso, Dkbeta, lam / rho; dofit = false)
    alpha = coef(fl)
    alpha .= 0
    numer = zeros(n)

    # Prepare the weight matrix.
    wt = if typeof(wt) <: AbstractVector && length(wt) == n
        Diagonal(wt)
    elseif typeof(wt) <: AbstractMatrix && all(size(wt) .== [n, n])
        wt
    else
        I(n)
    end

    # Adjust the ADMM parameter if needed to produce a non-singular problem.
    denom = nothing
    while true
        try
            denom = cholesky(wt + rho * DktDk)
            break
        catch
            rho /= 2
        end
    end

    work2 = zeros(n - order - 1)
    return Trendfilter(
        x,
        y,
        wt,
        beta,
        u,
        lam,
        rho,
        Dk,
        Dkp1,
        DktDk,
        Dkbeta,
        fl,
        numer,
        denom,
        work2,
        -1,
        false,
    )
end

function update!(tf::Trendfilter)

    (; x, y, beta, u, lam, rho, Dk, Dkp1, DktDk, Dkbeta, fl, numer, denom, work2) = tf

    apu = Dkbeta # alias to reuse memory

    # Equations 9-11 of Ramdas and Tibshirani
    alpha = coef(fl)
    apu .= alpha + u
    numer .= Dk' * apu
    numer .*= rho
    numer .+= y
    beta .= denom \ numer
    Dkbeta .= Dk * beta
    u .= Dkbeta - u
    fit!(fl, u, lam / rho)
    u .= alpha - u

    numer .= y - beta
    work2 .= Dkp1 * beta
    return sum(abs2, numer) / 2 + lam * sum(abs, work2)
end

function StatsBase.fit!(tf::Trendfilter; maxiter::Int = 600, tol::Float64 = 1e-6)

    oldobj = Inf
    tf.iter = -1
    tf.converged = false
    for itr = 1:maxiter
        obj = update!(tf)
        r = abs((obj - oldobj) / obj)
        if itr > 1 && r < tol
            tf.iter = itr
            tf.converged = true
            break
        end
        oldobj = obj
    end
end

function differencemats(x::Vector) where {T<:Real}

    n = length(x)

    D1 = BandedMatrix(Ones(n - 1, n), (0, 1))
    for i = 1:n-1
        D1[i, i] = -1 / (x[i+1] - x[i])
        D1[i, i+1] = 1 / (x[i+1] - x[i])
    end

    D2 = BandedMatrix(Ones(n - 2, n), (0, 2))
    for i = 1:n-2
        D2[i, i] = 2 / ((x[i+1] - x[i]) * (x[i+2] - x[i]))
        D2[i, i+1] = -2 / ((x[i+2] - x[i+1]) * (x[i+1] - x[i]))
        D2[i, i+2] = 2 / ((x[i+2] - x[i+1]) * (x[i+2] - x[i]))
    end

    D3 = BandedMatrix(Ones(n - 3, n), (0, 3))
    for i = 1:n-3
        D3[i, i] = 6 / ((x[i] - x[i+3]) * (x[i] - x[i+2]) * (x[i] - x[i+1]))
        D3[i, i+1] = -6 / ((x[i] - x[i+1]) * (x[i+1] - x[i+2]) * (x[i+1] - x[i+3]))
        D3[i, i+2] = 6 / ((x[i+1] - x[i+2]) * (x[i] - x[i+2]) * (x[i+2] - x[i+3]))
        D3[i, i+3] = -6 / ((x[i+2] - x[i+3]) * (x[i+1] - x[i+3]) * (x[i] - x[i+3]))
    end

    return D1, D2, D3
end
