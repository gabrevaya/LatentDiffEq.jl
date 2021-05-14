module LatentDiffEq

using OrdinaryDiffEq
using DiffEqFlux
using DiffEqSensitivity
using Flux
using Flux: reset!
using Statistics
using ModelingToolkit

## Defining types
abstract type AbstractModel end
abstract type AbstractEncoder end
abstract type AbstractDecoder end

abstract type LatentDE end

## Model definitions
include("./models/LatentDiffEqModel.jl")
include("./models/GOKU.jl")
include("./models/LatentODE.jl")
export LatentDiffEqModel, GOKU, LatentODE
export default_layers

## Predefined systems
include("./systems/pendulum.jl")
include("./systems/nODE.jl")
export Pendulum, NODE

include("./utils/utils.jl")
export kl, vector_mse, annealing_factor
export normalize_to_unit_segment, time_loader

end # end LatentDiffEq module