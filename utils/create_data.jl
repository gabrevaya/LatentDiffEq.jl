

using OrdinaryDiffEq
using BSON:@save, @load
using BSON
using Parameters: @with_kw
using Random
using Statistics
using Distributions
using ModelingToolkit
using Flux
using LinearAlgebra

abstract type AbstractSystem end
include("../system/lv_problem.jl")
include("../system/full_vdP.jl")


## ARGUMENTS IN THIS STRUCT MUSH BE THE SAME AS THE ONE IN GOKU_TRAIN.JL
@with_kw mutable struct Args_gen

    ## Dynamical system
    system = vdP(5)               # Available : "LV(), vdP(k)"

    ## Mask dimensions
    input_dim = 10               # model input size
    hidden_dim_gen = 10         # hidden dimension of the g function

    ## time and parameter ranges
    full_t_span = (0.0, 19.95)  # full time span of training exemple (un-sequenced)
    dt = 0.05                   # timestep for ode solve
    u₀_range = (1.5, 3.0)       # initial value range
    p₀_range = (1.0, 2.0)       # parameter value range

    ## Save paths and keys
    data_file_name = "vdP_data.bson"  # data file name
    seed = 1                         # random seed

end

function generate_dataset(; kws...)

      args = Args_gen(; kws...)
      Random.seed!(args.seed)

      ##########################################################################
      ## Problem definition

      prob = ODEProblem(args.system.f!, args.system.u₀, args.full_t_span, args.system.p, jac=true, sparse=true)

      ##########################################################################
      ## Function definition

      output_func(sol, i) = (Array(sol), false)
      rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
      rand_uniform(range::Array, size) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]

      # for samping initial condition from a uniform distribution
      function prob_func(prob,i,repeat)
            u0_new = rand_uniform(args.u₀_range, length(prob.u0))
            p_new = rand_uniform(args.p₀_range, length(prob.p))
            prob = remake(prob; u0 = u0_new, p = p_new)
      end

      function prob_func_2(prob,i,repeat)
            u0_new = rand_uniform(args.u₀_range, length(prob.u0))
            prob = remake(prob; u0 = u0_new)
      end

      ##########################################################################
      ## Create data

      # Solve for X trajectories
      ensemble_prob = EnsembleProblem(prob, prob_func=prob_func_2, output_func = output_func)
      sim = solve(ensemble_prob, Tsit5(), saveat=args.dt, trajectories=10000)

      raw_data = dropdims(Array(sim), dims = 2)

      # Probably works but requieres alot of RAM for some reason
      # When solving this issue, add include("utils.jl")
      # norm_data = zeros(Float32, size(raw_data))
      # norm_data = normalize_Z(raw_data)

      mask = gen(args.system, args.hidden_dim_gen, args.input_dim)
      data_masked = mask.(Flux.unstack(raw_data, 2))
      data_masked = Flux.stack(data_masked, 2)

      @save args.data_file_name raw_data data_masked
end

function solve_prob(u0, p, tspan, tstep)

      @parameters t α β δ γ
      @variables x(t) y(t)
      @derivatives D'~t

      u₀ = [x => u0[1]
            y => u0[2]]
      p = [ α => p[1]
            β => p[2]
            δ => p[3]
            γ => p[4]]

      eqs = [D(x) ~ α*x - β*x*y,
           D(y) ~ -δ*y + γ*x*y]

      sys = ODESystem(eqs)
      prob = ODEProblem(sys, u₀, tspan, p, jac=true, sparse=true)
      sol = solve(prob, Vern7(), saveat = tstep)

      sol
end
