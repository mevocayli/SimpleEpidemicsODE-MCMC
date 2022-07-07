using DifferentialEquations , LinearAlgebra, Plots, StatsPlots, Distributions, Random, SparseArrays





function sir_ode(du,u,p,t)
    S,E,I,R = u
    beta,alpha,c1,c2,DI = p
    du[1] = R/DI - S*beta*(E+I/S+E+I+R)
    du[2] = S*beta*(E+I/S+E+I+R) - E*alpha
    du[3] = E*alpha - I*c1
    du[4] = I*c1 + E*c2 - R/DI
end


const u0 = [19000.0,1000.0,1.0,0.0]
const tspan = (0.0,10.0)



function metropolis(S::Int64, width::Float64, width2::Float64, ρ::Float64;
                    start_alpha::Float64=0.1,start_beta::Float64=0.1,start_c1::Float64=0.1,start_c2::Float64=0.3,start_DI::Float64=2.0,
                    σ_alpha::Float64=0.01, σ_beta::Float64=0.01,σ_c1::Float64=0.01,σ_c2::Float64=0.01,σ_DI::Float64=0.5,
                    seed=123,n=10::Int64)
    rgn = MersenneTwister(seed)
    data = Float64[1000.0,1700.0,1720.0,1730.0,1780.0,1740.0,1800.0,1950.0,1900.0,2200.0,2300.0]
    draws = Matrix{Float64}(undef, S, 5)
    accepted = 0::Int64;
    beta = start_beta; alpha = start_alpha; c1 = start_c1; c2 = start_c2; DI = start_DI
    @inbounds draws[1, :] = [beta alpha c1 c2 DI]
    for s in 2:S
        beta_ = rand(rgn, Uniform(beta - width2, beta + width))
        alpha_ = rand(rgn, Uniform(alpha - width2, alpha + width))
        c1_ = rand(rgn, Uniform(c1 - width, c1 + width))
        c2_ = rand(rgn, Uniform(c2 - width, c2 + width))
        DI_ = rand(rgn, Uniform(DI - width2, DI + width2))
        p_ = [beta_,alpha_,c1_,c2_,DI_]
        prob = ODEProblem(sir_ode,u0,tspan,p_)
        sol1 = solve(prob, saveat = 1.0)
        odedata = Array(sol1)
        estimated = odedata[2,:]
        r = (1/2*pi*σ_alpha)^n/2*exp(-0.5*σ_alpha^-2*sum(data[:]-estimated[:])^2)

        if r > rand(rgn, Uniform())
            beta = beta_
            alpha = alpha_
            c1 = c1_
            c2 = c2_
            DI = DI_
            accepted += 1
        end
        @inbounds draws[s, :] = p_
    end
    println("Acceptance rate is: $(accepted / S)")
    return draws
end



const S = 10_000
const width = 0.01
const width2 = 1.0
const ρ = 0.8

X_met = metropolis(S, width,width2, ρ)
