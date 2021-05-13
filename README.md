# LatentDiffEq [![Build Status](https://travis-ci.com/gabrevaya/LatentDiffEq.jl.svg?branch=master)](https://travis-ci.com/gabrevaya/LatentDiffEq.jl)

Latent Differential Equation models in Julia.

#### Note: This repo is still experimental!

## Installation

1. Clone the repository and enter the folder:
```
$ git clone git@github.com:gabrevaya/LatentDE.git
$ cd LatentDE
```
2. Instantiate and try the GOKU-net model on a pendulum example
```julia
$ julia
julia> ] activate .
julia> ] instantiate
julia> include("LatentDiffEq.jl")
julia> include("pendulum_friction-less/model_train.jl")
julia> train()
```

## References

Linial, O., Eytan, D., & Shalit, U. (2020). Generative ODE Modeling with Known Unknowns. arXiv preprint [arXiv:2003.10775](https://arxiv.org/abs/2003.10775).