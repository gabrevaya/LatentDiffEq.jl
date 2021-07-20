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

@with_kw mutable struct Args_gen
      ## Latent Differential Equations
      diffeq = DoublePendulum()
  
      tspan = (0.0f0, 5f0)        # time span
      dt = 0.05                       # timestep for ode solve
      u₀_range = (-π/3, π/3)          # initial value range
      p₀_range = (1.0, 2.0)           # parameter value range
      n_traj = 450                    # Number of trajectories
      seed = 1                        # random seed

      ## High dimensional data arguments
      high_dim_args = (10, 1, 2)    # for the the simple pendulum
                                      # (pendulumlength, radius, rodthickness)
end

function generate_dataset(; kws...)
      args = Args_gen(; kws...)
      @unpack_Args_gen args
      Random.seed!(seed)

      ## Problem definition
      prob = remake(diffeq.prob, tspan = tspan)

      ## Ensemble functions definition
      prob_func(prob,i,repeat) = remake(prob, u0 = u0s[i], p = ps[i])

      # Sample initial condition and parameters from a uniform distribution
      ps = [rand_uniform(p₀_range, length(prob.p)) for i in 1:n_traj]
      u0s = [rand_uniform(u₀_range, length(prob.u0)) for i in 1:n_traj]
      # u0s = [vcat(rand_uniform_u0(u₀_range, length(prob.u0)), 0) for i in 1:n_traj]

      @info "Generating data"
      ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
      sols = solve(ensemble_prob, diffeq.solver, saveat = args.dt, trajectories = n_traj)

      latent_data = [dropdims(Array(sol), dims = 2) for sol in sols]
      high_dim_data = [create_high_dim_data(sol, high_dim_args) for sol in sols]

      return latent_data, u0s, ps, high_dim_data
end

## util functions
rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
rand_uniform(range::Array, size) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]
rand_uniform_u0(range::Tuple, size) = rand(Uniform(range...),(Int(size/2),1))

## Luxor util functions
# convert a timeseries to two trajectories

# convert a timeseries to two trajectories
function maketrajectories(tseries, pendulumlength)
      trajectories = Tuple{Point,Point}[]
      @inbounds for i in 1:length(tseries)
          angle1 = π/2 + tseries[i][1]
          angle2 = π/2 + tseries[i][3]
          pos1 = Point(pendulumlength * cos(angle1), pendulumlength * sin(angle1))
          pos2 = pos1 + Point(pendulumlength * cos(angle2), pendulumlength * sin(angle2))
          push!(trajectories, (pos1, pos2))
      end
      return trajectories
end
  
function drawrod(startpos, endpos, linewidth)
      rodlength = distance(startpos, endpos)
      setline(linewidth)
      line(startpos, endpos, :stroke)
end

function frame(pos, radius, rodthickness)
      innerpos, outerpos = pos
      background("black")
      sethue("white")
      drawrod(O,         innerpos, rodthickness)
      drawrod(innerpos,  outerpos, rodthickness)
      circle(O, radius, :fill)
      circle(innerpos, radius, :fill)
      circle(outerpos, radius, :fill)
      sethue("black")
      circle(O, radius/2, :fill)
      circle(innerpos, radius/2, :fill)
end

function create_frames(trajectories, radius, rodthickness, w=50, h=50)
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

latent_data, u0s, ps, high_dim_data = generate_dataset()

root_dir = @__DIR__
mkpath("$root_dir/data")
data_file_name = "$root_dir/data/data.bson"

data = (latent_data, u0s, ps, high_dim_data)
@save data_file_name data