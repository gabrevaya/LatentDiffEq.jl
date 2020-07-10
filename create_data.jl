using ModelingToolkit
using OrdinaryDiffEq
using Random
using Distributions
using BSON: @save, @load

function generate_dataset()

      @parameters t α β δ γ
      @variables x(t) y(t)
      @derivatives D'~t

      eqs = [D(x) ~ α*x - β*x*y,
             D(y) ~ -δ*y + γ*x*y]

      sys = ODESystem(eqs)

      u0 = [x => 1.0
            y => 1.0]

      p  = [α => 1.5
            β => 1.0
            δ => 3.0
            γ => 1.0]

      tspan = (0.0, 9.95)
      prob = ODEProblem(sys,u0, tspan, p, jac=true, sparse=true)
      sol = solve(prob, Vern7(), saveat = 0.1)

      p₀ = [pᵢ[2] for pᵢ ∈ p]
      u₀ = [u₀ᵢ[2] for u₀ᵢ ∈ u0]
      # u₀_range = [0.2 2.0
      #             0.2 2.0]

      u₀_range = (1.5, 3.0)
      p₀_range = (1.0, 2.0)

      # for samping initial condition from a uniform distribution
      function prob_func(prob,i,repeat)
            u0_new = rand_uniform(u₀_range, length(prob.u0))
            remake(prob, u0 = u0_new)
      end

      # for samping initial condition from a normal distribution
      function prob_func2(prob,i,repeat)
            u0_new  = u₀ + 0.1*randn(length(u₀))
            remake(prob, u0 = u0_new)
      end

      output_func(sol,i) = (Array(sol),false)
      rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
      rand_uniform(range::Array, size) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]


      # generate some fixed random parameters
      p_new = rand_uniform(p₀_range, length(prob.p))
      # p_new = p₀ + 0.1*randn(length(p₀))
      remake(prob, p = p_new)

      ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, output_func = output_func)
      # ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
      sim = solve(ensemble_prob, Vern7(), saveat=0.1, trajectories=10000)

      full_data = dropdims(Array(sim), dims = 2)

      @save "lv_data.bson" full_data p
end


function solve_prob(u0, p, tspan, tstep)

      @parameters t α β δ γ
      @variables x(t) y(t)
      @derivatives D'~t

      u₀ = [x => u0[1]
            y => u0[2]]

      eqs = [D(x) ~ α*x - β*x*y,
           D(y) ~ -δ*y + γ*x*y]

      sys = ODESystem(eqs)
      prob = ODEProblem(sys, u₀, tspan, p, jac=true, sparse=true)
      sol = solve(prob, Vern7(), saveat = tstep)

      sol
end
