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
include("./systems/pendulum.jl")
export pendulum
include("./systems/Kuramoto.jl")
export Kuramoto, Kuramoto_basic
include("./systems/Hopf.jl")
export Hopf
include("./systems/Lotka-Volterra.jl")
export LV
include("./systems/van_der_Pol.jl")
export vdP_full, vdP_identical_local
include("./systems/Wilson-Cowan.jl")
export WC_full, WC, WC_identical_local
include("./systems/Stochastic_van_der_Pol.jl")
export SvdP_full
include("./systems/Stochastic_Lotka-Volterra.jl")
export SLV

## Model definitions
include("./models/GOKU.jl")
export Goku

end # end LatentDE module