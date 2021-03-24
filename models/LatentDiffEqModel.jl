
struct LatentDiffEqModel{T,E,D} <: AbstractModel

    model_type::T
    encoder::E
    decoder::D

    function LatentDiffEqModel(model_type, encoder_layers, diffeq, decoder_layers)

        encoder = built_encoder(model_type, encoder_layers)
        decoder = built_decoder(model_type, decoder_layers, diffeq)
        T, E, D = typeof(model_type), typeof(encoder), typeof(decoder)
        new{T, E, D}(model_type, encoder, decoder)
    end
end

function (model::LatentDiffEqModel)(x,t)

    ## Get encoded latent initial states and parameters
    μ, logσ² = model.encoder(x)

    # Sample from distributions
    l̃ = variational(model.model_type, μ, logσ²)

    ## Get predicted output
    X̂ = model.decoder(l̃, t)

    return X̂, μ, logσ²
end

# default non variational
# variational(model, μ, logσ²) = _variational(model.model_type, μ, logσ²)
# _variational(model_type, μ, logσ²) = μ
variational(model_type, μ, logσ²) = μ

Flux.@functor LatentDiffEqModel