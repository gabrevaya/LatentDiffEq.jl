

using OrdinaryDiffEq
using BSON:@save, @load
using BSON
using Parameters: @with_kw
using Random
using Statistics
using Distributions
using ModelingToolkit
using Flux

abstract type AbstractSystem end
include("utils.jl")
include("../system/Lotka-Volterra.jl")
include("../system/van_der_Pol.jl")
include("../system/Wilson-Cowan.jl")
include("../system/Kuramoto.jl")
include("../system/Hopf.jl")


## ARGUMENTS IN THIS STRUCT MUSH BE THE SAME AS THE ONE IN GOKU_TRAIN.JL
@with_kw mutable struct Args_gen

    ## Dynamical system
    system = Kuramoto(2)        # Available : LV(), vdP_full(k),
                                #             vdP_identical_local(k)
                                #             WC_full(k), WC(k),
                                #             WC_identical_local(k)
                                #             Kuramoto_basic(k), Kuramoto(k)
                                #             Hopf(k)
                                #             (k → number of oscillators)

    ## Mask dimensions
    input_dim = 4               # model input size
    hidden_dim_gen = 10         # hidden dimension of the g function

    ## time and parameter ranges
    full_t_span = (0.0f0, 19.95f0)  # full time span of training exemple (un-sequenced)
    dt = 0.05                   # timestep for ode solve
    u₀_range = (1.5, 3.0)       # initial value range
    p₀_range = (1.0, 2.0)       # parameter value range

    ## Save paths and keys
    data_file_name = "kuramoto_data.bson"  # data file name
    seed = 1                         # random seed

end

function generate_dataset(; kws...)

      args = Args_gen(; kws...)
      Random.seed!(args.seed)

      ##########################################################################
      ## Problem definition

      prob = remake(args.system.prob, tspan = args.full_t_span)

      ##########################################################################
      ## Function definition

      output_func(sol, i) = (Array(sol), false)
      rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1)) # TODO make it preserve the type
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
            # prob = remake(prob; u0 = SArray{Tuple{size(prob.u0)...}}(u0_new))
      end

      ##########################################################################
      ## Create data

      @info "Creating data"
      # Solve for X trajectories
      ensemble_prob = EnsembleProblem(prob, prob_func=prob_func_2, output_func = output_func)
      sim = solve(ensemble_prob, Tsit5(), saveat=args.dt, trajectories=10000)
      raw_data = dropdims(Array(sim), dims = 2)
      transformed_data = args.system.transform(raw_data)

      # Probably works but requieres alot of RAM for some reason
      # When solving this issue, add include("utils.jl")
      # norm_data = zeros(Float32, size(raw_data))
      # norm_data = normalize_Z(raw_data)

      @info "Applying mask"
      mask = gen(length(args.system.u₀), args.hidden_dim_gen, args.input_dim)
      data_masked = mask.(Flux.unstack(raw_data, 2))
      data_masked = Flux.stack(data_masked, 2)

      @info "Saving data"
      @save args.data_file_name raw_data transformed_data data_masked
end

function transform_dataset(raw_data, window_size=400, interval=5)
      start_idx = collect(1:interval:size(raw_data)[2]-window_size)
      transformed_data = zeros(size(raw_data)[1], window_size, length(start_idx))
      for i in 1:length(start_idx)
            idx = start_idx[i]
            transformed_data[:,:,i] = raw_data[:,idx:idx+window_size-1]
      end
      return transformed_data
end

# Deterministic neural-net used to get from state to imput sample for the GOKU architecture (input_dim ≂̸ ode_dim)
gen(raw_in, hidden_dim_gen, input_dim) = Chain(Dense(raw_in, hidden_dim_gen, relu),
                                                   Dense(hidden_dim_gen, input_dim, relu))
generate_dataset()
