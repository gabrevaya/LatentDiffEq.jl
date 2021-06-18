using LatentDiffEq
using OrdinaryDiffEq
using BSON:@save
using Parameters: @with_kw
using Random
using Statistics
using Distributions
using ModelingToolkit
using Flux
using Luxor
using Images

include("FreeFall.jl")

@with_kw mutable struct Args_gen
      ## Latent Differential Equations
      # diffeq = Pendulum()
  
      tspan = (0.0f0, 4.95f0)         # time span
      dt = 0.05f0                       # timestep for ode solve
      u₀_range = (9.f0, 11.f0)          # initial value range
      p₀_range = (-3.0f0, -0.5f0)          # parameter value range
      n_traj = 250                    # Number of trajectories changed from 450
      seed = 1                        # random seed
  
      ## High dimensional data arguments
      high_dim_args = (2.5)             # for the free fall
                                         # (radius of the ball)
end

function generate_dataset(; diffeq = FreeFall(), kws...) 
      args = Args_gen(; kws...)
      @unpack_Args_gen args
      Random.seed!(seed)

      ## Problem definition
      prob = remake(diffeq.prob, tspan = tspan)

      ## Ensemble functions definition
      prob_func(prob,i,repeat) = remake(prob, u0 = u0s[i], p = ps[i])

      # Sample initial condition and parameters from a uniform distribution
      ps = [rand_uniform(p₀_range, length(prob.p)) for i in 1:n_traj]
      # u0s = [rand_uniform(u₀_range, length(prob.u0)) for i in 1:n_traj] #replace with below
      u0s = [vcat(rand_uniform_u0(u₀_range, length(prob.u0)), 0) for i in 1:n_traj]

      # Build and solve EnsembleProblem data
      ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
      sols = solve(ensemble_prob, diffeq.solver, saveat = args.dt, trajectories = n_traj)

      # Store the trajectories variables
      latent_data = [dropdims(Array(sol), dims = 2) for sol in sols]

      # Create animations with Luxor
      high_dim_data = [create_high_dim_data(sol, high_dim_args) for sol in sols]

      return latent_data, u0s, ps, high_dim_data
end

## util functions
rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
rand_uniform(range::Array, size) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]
rand_uniform_u0(range::Tuple, size) = rand(Uniform(range...),(Int(size/2),1))


## Luxor util functions
# convert a timeseries to two trajectories

function maketrajectories(tseries)
      innertrajectory = Point[]
      @inbounds for i in 1:length(tseries)
            angle1 = tseries[i][1]
            pos1 = Point(0, tseries[i][1][1])
            push!(innertrajectory, pos1)
      end
      return innertrajectory
end

function frame(pos, radius)
      background("black")
      sethue("white")
      offset = Point(0,-8.5)
      pos = -pos
      # pos += offset 
      circle(pos, radius, :fill)

end

function create_frames(trajectories, radius, w=28, h=28)
      frames = Matrix{Float32}[]
      buffer = zeros(UInt32, w, h)
      for pos in trajectories
          a = @imagematrix! buffer frame(pos, radius) w h
          push!(frames, Float32.(Gray.(a)))
      end
      frames
end

function create_high_dim_data(sol, high_dim_args)
      radius = high_dim_args
      trajectories = maketrajectories(sol)
      create_frames(trajectories, radius)
end
