module LatentDiffEq

using OrdinaryDiffEq
using DiffEqFlux
using DiffEqSensitivity
using Flux
using Flux: reset!
using Statistics
using ModelingToolkit
using DynamicalSystems
using StochasticDiffEq

## Defining types
abstract type LatentDE end

## Model definitions
include("./models/LatentDiffEqModel.jl")
include("./models/GOKU.jl")
include("./models/LatentODE.jl")
export LatentDiffEqModel, GOKU, LatentODE
export default_layers

## Predefined systems
include("./systems/pendulum.jl")
include("./systems/pendulum_NN_friction.jl")
include("./systems/double_pendulum.jl")
include("./systems/nODE.jl")

export Pendulum, SPendulum, DoublePendulum, NODE
export Pendulum_friction, Pendulum_NN_friction

include("./utils/utils.jl")
export vector_mse, kl, vector_kl, annealing_factor
export normalize_to_unit_segment, time_loader, rand_time

end # end LatentDiffEq module
