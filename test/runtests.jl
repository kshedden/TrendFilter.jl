using TrendFilter, StableRNGs, Test, Statistics, LinearAlgebra

@testset "Difference matrices" begin

    n = 20
    rng = StableRNG(123)
    x = randn(rng, n)
    sort!(x)

    D1, D2, D3 = TrendFilter.differencemats(x)

    # Test a polynomial of degree j.
    for j = 1:3

        y = zeros(n) .+ randn()
        c = randn(rng, j)
        for k in eachindex(c)
            y = y + c[k] * x .^ k
        end

        dd = [D1 * y, D2 * y, D3 * y]
        dn = norm.(dd)

        for k = j+1:3
            @test isapprox(dn[k], 0, atol = 1e-8, rtol = 1e-8)
        end
        @test isapprox(std(dd[j]), 0, atol = 1e-8, rtol = 1e-8)

        f = first(dd[j]) / c[j]
        @test isapprox(f, prod(1:j))
    end
end

@testset "Example 1" begin

    rng = StableRNG(123)
    n = 1000

    for j = 1:10
        x = randn(rng, n)
        sort!(x)
        ey = x .^ 2
        ey[x.<=0] .= 0
        y = ey + randn(rng, n)

        tf = fit(Trendfilter, x, y, 1.0; order = 1)
        yh = coef(tf)

        @test mean(abs, ey - yh) < 0.1
    end
end
