module LatentDE

using OrdinaryDiffEq
using BSON:@save, @load
using DiffEqFlux
using DiffEqSensitivity
using Flux
using Flux: reset!
using Statistics
using Zygote
using ModelingToolkit
using Images
using Plots

## Defining types
abstract type AbstractEncoder end
abstract type AbstractDecoder end
abstract type AbstractModel end
abstract type AbstractSystem end

include("./utils/utils.jl")
export KL, loss_batch, annealing_factor
export NormalizeToUnitSegment, time_loader
export create_prob

include("./utils/visualize.jl")
export visualize_val_image

## Predefined systems
include("./system/pendulum.jl")
export pendulum

include("./system/Kuramoto.jl")
export Kuramoto

## Model definitions
include("./model/GOKU_model_video2.jl")
export Goku

end # end LatentDE module