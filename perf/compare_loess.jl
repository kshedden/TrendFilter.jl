using TrendFilter, Loess, Printf, TimerOutputs

n = 1000
nrep = 100
order = 2

function sim(f, x)
    mae_tf, mae_loess = 0.0, 0.0
    to = TimerOutput()

    lf = (x, y) -> begin
        m = loess(x, y)
        predict(m, x)
    end
    tf = (x, y) -> begin
        m = Trendfilter(x, y, 1.0; order = order)
        fit!(m)
        coef(m)
    end

    for ii = 1:nrep

        ey = f.(x)
        y = ey + randn(n)

        if ii == 1
            @timeit to "tf" tf(x, y)
            @timeit to "loess" lf(x, y)
        end

        yh = tf(x, y)
        mae_tf += sum(abs, yh - ey) / n

        yl = lf(x, y)
        mae_loess += sum(abs, yl - ey) / n
    end

    show(to)
    println("")

    mae_tf /= nrep
    mae_loess /= nrep
    println(@sprintf("MAE TF=%5.3f MAE Loess=%5.3f", mae_tf, mae_loess))
    println("")
end

println("Quadratic:")
x = randn(n)
sort!(x)
sim(x -> x^2, x)

println("Sine wave (1 cycle):")
x = 2 * pi * rand(n) .- 1 * pi
sort!(x)
sim(x -> sin(x), x)

println("Sine wave * I(x > 0) (1 cycle)")
x = 2 * pi * rand(n) .- 1 * pi
sort!(x)
sim(x -> x > 0 ? sin(x) : 0.0, x)

println("Sine wave (2 cycles)")
x = 4 * pi * rand(n) .- 2 * pi
sort!(x)
sim(x -> sin(x), x)
