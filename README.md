# LatentDiffEq.jl 

[![Build Status](https://travis-ci.com/gabrevaya/LatentDiffEq.jl.svg?branch=master)](https://travis-ci.com/github/gabrevaya/LatentDiffEq.jl)

Latent Differential Equation models is a powerful class of models that brings great promise to the field of time series processing. [Flux.jl](https://github.com/FluxML/Flux.jl) and the [SciML](https://github.com/SciML) provide all the tools to build these latent DE models. However, the process of setting up those models can be a time-consuming endeavor with a quite steep learning curve for people without a background in machine learning. LatentDiffEq.jl aims to provide a framework that makes latent differential equation models readily accessible to people that needs it most. This is done by providing pre-programmed established models in Latent Differential Equations while offering a high level of flexibility for new user-defined models.

![LDE_framework](https://user-images.githubusercontent.com/19957518/128905421-c36ac189-77e0-4df2-b7f0-6d7b50ff8158.png)

This package is still in an early (but functional) stage of its development, any help in finding bugs or improvement would be greatly appreciated. Please get in touch if, like us, you share a great interest in bringing this exciting field of research to a wider audience!

## Getting Started
### Installation

```julia
julia> ]
pkg> add https://github.com/gabrevaya/LatentDiffEq.jl.git
```

### Tutorial
Checkout the [tutorial](./examples/tutorial/GOKU-net_pendulum_tutorial.ipynb) to get familiar with the base functionalities of this package. You will also learn how to build your own models within our framework. In addition, you can refer to the following chart detailing the names of the functions associated with the different parts of the package.

![LatentDiffEq.jl_framework](https://user-images.githubusercontent.com/19957518/128906143-41dd1d0a-d081-4261-b413-0327ad5eace2.png)

## References

Chen, Ricky TQ, et al. "Neural ordinary differential equations." arXiv preprint [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) (2018).

Linial, Ori, et al. "Generative ODE modeling with known unknowns." [Proceedings of the Conference on Health, Inference, and Learning.](https://dl.acm.org/doi/abs/10.1145/3450439.3451866) (2021).