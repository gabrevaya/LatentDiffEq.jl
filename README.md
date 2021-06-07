# LatentDiffEq [![Build Status](https://api.travis-ci.com/gabrevaya/LatentDiffEq.svg?branch=master)](https://travis-ci.com/github/gabrevaya/LatentDiffEq)

Latent Differential Equations models in Julia.

#### Note: This repo is still experimental!

## GOKU-net model on a pendulum example
1. Clone the repository and enter the folder:
```
$ git clone https://github.com/gabrevaya/LatentDiffEq.git
$ cd LatentDiffEq/examples/pendulum_friction-less
```
2. Activate and instantiate the project
```julia
$ julia
julia> ]
pkg> activate .
pkg> instantiate
```
3. Run the training script
```julia
julia> include("model_train.jl")
```

## Installation

```julia
julia> ]
pkg> add https://github.com/gabrevaya/LatentDiffEq.git
```


## References

Linial, O., Eytan, D., & Shalit, U. (2020). Generative ODE Modeling with Known Unknowns. arXiv preprint [arXiv:2003.10775](https://arxiv.org/abs/2003.10775).
