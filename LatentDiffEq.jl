module LatentDiffEq

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
include("./systems/Kuramoto.jl")
include("./systems/nODE.jl")
# include("./systems/Hopf.jl")
# include("./systems/Lotka-Volterra.jl")
# include("./systems/van_der_Pol.jl")
# include("./systems/Wilson-Cowan.jl")
# export SLV, Pendulum, Kuramoto, Kuramoto_basic, Hopf
# export LV, vdP_full, vdP_identical_local, WC_full
# export WC, WC_identical_local, SvdP_full, Stoch_Hopf
export Pendulum, Kuramoto_full, Kuramoto_basic, NODE

include("./utils/utils.jl")
export kl, loss_batch, annealing_factor
export normalize_to_unit_segment, time_loader
export visualize_val_image

end # end LatentDiffEq module