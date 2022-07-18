using TrendFilter, UnicodePlots, Loess

n = 2000
nrep = 100

function f()
    mae_tf = 0.0
    mae_loess = 0.0
    for ii = 1:nrep

        x = 4 * pi * rand(n) .- 2 * pi
        sort!(x)
        ey = sin.(x)
        ey[ey.<0] .= 0
        y = ey + randn(n)

        tf = Trendfilter(x, y, 2.0)
        fit!(tf)
        mae_tf += sum(abs, coef(tf) - ey) / n

        m = loess(x, y)
        yl = predict(m, x)
        mae_loess += sum(abs, yl - ey) / n
    end

    mae_tf /= nrep
    mae_loess /= nrep
    println(mae_tf, " ", mae_loess)
end

f()


#println(tf.iter)

#plt = scatterplot(ey, coef(tf))
#println(plt)
