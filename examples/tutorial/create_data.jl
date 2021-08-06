using LatentDiffEq
using OrdinaryDiffEq
using BSON:@save
using Parameters: @with_kw
using Random
using Statistics
using Distributions
using Flux
using Luxor
using Images


@with_kw mutable struct Args_gen
      ## Latent Differential Equations
      # diffeq = Pendulum()
  
      tspan = (0.0f0, 4.95f0)         # time span
      dt = 0.05                       # timestep for ode solve
      u₀_range = [-π/6 π/6            # initial values ranges
                  -π/3 π/3]

      p₀_range = (1.0, 2.0)           # parameter value range
      n_traj = 450                    # Number of trajectories
      seed = 1                        # random seed
  
      ## High dimensional data arguments
      high_dim_args = (19, 1.75, 3.75)    # for the the simple pendulum
                                         # (pendulumlength, radius, rodthickness)
end

function generate_dataset(; diffeq = Pendulum(), kws...)
      args = Args_gen(; kws...)
      @unpack_Args_gen args
      Random.seed!(seed)

      ## Problem definition
      prob = remake(diffeq.prob, tspan = tspan)

      ## Ensemble functions definition
      prob_func(prob,i,repeat) = remake(prob, u0 = u0s[i], p = ps[i])

      # Sample initial condition and parameters from a uniform distribution
      ps = [rand_uniform(p₀_range, length(prob.p)) for i in 1:n_traj]
      u0s = [rand_uniform(u₀_range) for i in 1:n_traj]

      # Build and solve EnsembleProblem data
      ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
      sols = solve(ensemble_prob, diffeq.solver, saveat = args.dt, trajectories = n_traj)

      # Store the trajectories variables
      latent_data = [Array(sol) for sol in sols]

      # Create animations with Luxor
      high_dim_data = [create_high_dim_data(sol, high_dim_args) for sol in sols]

      return latent_data, u0s, ps, high_dim_data
end

## util functions
rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
rand_uniform(range::Array) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]
rand_uniform_u0(range::Tuple, size) = rand(Uniform(range...),(Int(size/2),1))


## Luxor util functions
# convert a timeseries to two trajectories
function maketrajectories(tseries, pendulumlength)
      innertrajectory = Point[]
      @inbounds for i in 1:length(tseries)
            angle1 = π/2 + tseries[i][1]
            pos1 = Point(pendulumlength * cos(angle1), pendulumlength * sin(angle1))
            push!(innertrajectory, pos1)
      end
      return innertrajectory
end

function drawrod(startpos, endpos, linewidth)
      rodlength = distance(startpos, endpos)
      setline(linewidth)
      line(startpos, endpos, :stroke)

      p = (startpos + endpos) / 2
      θ = atan(p[2] - startpos[2], p[1] - startpos[1])
      sethue("black")
      fontsize(8)
      # text("julia", p, valign=:middle, halign=:center, angle = θ)
      Luxor.text("|", p, valign=:middle, halign=:center, angle = θ)
end

function frame(pos, radius, rodthickness)
      background("black")
      sethue("white")
      offset = Point(0,-8.5)
      pos += offset
      sethue("white")
      circle(pos, radius, :fill)
      circle(offset, radius, :fill)
      drawrod(offset, pos, rodthickness)
      sethue("black")
      circle(offset, radius/2, :fill)
end

function create_frames(trajectories, radius, rodthickness, w=28, h=28)
      frames = Matrix{Float32}[]
      buffer = zeros(UInt32, w, h)
      for pos in trajectories
          a = @imagematrix! buffer frame(pos, radius, rodthickness) w h
          push!(frames, Float32.(Gray.(a)))
      end
      frames
end

function create_high_dim_data(sol, high_dim_args)
      pendulumlength, radius, rodthickness = high_dim_args
      trajectories = maketrajectories(sol, pendulumlength)
      frames = create_frames(trajectories, radius, rodthickness)
end

# latent_data, u0s, ps, high_dim_data = generate_dataset()

# data = (latent_data, u0s, ps, high_dim_data)
# root_dir = @__DIR__
# mkpath("$root_dir/data")
# @save "$root_dir/data/data.bson" data