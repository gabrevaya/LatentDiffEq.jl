module LatentDE

using OrdinaryDiffEq
using BSON:@save, @load
using DiffEqFlux
using DiffEqSensitivity
using Flux
using Flux: reset!
using Statistics
using ModelingToolkit
using Images
using Plots

## Defining types
abstract type AbstractEncoder end
abstract type AbstractDecoder end
abstract type AbstractModel end
abstract type AbstractSystem end
export AbstractSystem

include("./utils/utils.jl")
export KL, loss_batch, annealing_factor
export NormalizeToUnitSegment, time_loader

include("./utils/visualize.jl")
export visualize_val_image

## Model definitions
include("./models/GOKU.jl")
include("./models/LatentODE.jl")
export Goku, LatentODE

## Predefined systems
include("./systems/pendulum.jl")
include("./systems/Kuramoto.jl")
include("./systems/Hopf.jl")
include("./systems/Lotka-Volterra.jl")
include("./systems/van_der_Pol.jl")
include("./systems/Wilson-Cowan.jl")
include("./systems/Stochastic_van_der_Pol.jl")
include("./systems/Stochastic_Lotka-Volterra.jl")
export SLV, pendulum, Kuramoto, Kuramoto_basic, Hopf
export LV, vdP_full, vdP_identical_local, WC_full
export WC, WC_identical_local, SvdP_full

end # end LatentDE module