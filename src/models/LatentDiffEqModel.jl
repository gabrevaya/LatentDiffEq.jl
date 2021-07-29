struct LatentDiffEqModel{M,E,D}

    model_type::M
    encoder::E
    decoder::D

    @doc raw"""
        LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

    Construct a LatentDiffEqModel.

    # Arguments
    model_type: GOKU() or LatentODE()
    encoder_layers, decoder_layers: Could be specified manually or produced from default_layers
    """
    function LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

        encoder = Encoder(model_type, encoder_layers)
        decoder = Decoder(model_type, decoder_layers)
        M, E, D = typeof(model_type), typeof(encoder), typeof(decoder)
        new{M, E, D}(model_type, encoder, decoder)
    end
end

function (model::LatentDiffEqModel)(x, t, variational=false)

    # Get encoded latent initial states and parameters
    μ, logσ² = model.encoder(x)

    # Sample from distributions
    l̃ = variational ? sample(μ, logσ², model) : μ

    # Get predicted output
    X̂ = model.decoder(l̃, t)

    return X̂, μ, logσ²
end

Flux.@functor LatentDiffEqModel

struct Encoder{M,FE,PE,LI}

    model_type::M

    feature_extractor::FE
    pattern_extractor::PE
    latent_in::LI

    @doc raw"""
        Encoder(model_type, encoder_layers)

    Construct an encoder of a LatentDiffEqModel.

    # Arguments
    model_type: GOKU() or LatentODE()
    encoder_layers: Could be specified manually or produced be default_layers
    """
    function Encoder(model_type, encoder_layers)
        M = typeof(model_type)
        FE, PE, LI = typeof.(encoder_layers)
        new{M,FE,PE,LI}(model_type, encoder_layers...)
    end
end

function (encoder::Encoder)(x)

    # Pass every time frame independently through the feature extractor
    fe_out = apply_feature_extractor(encoder, x)

    # Process sequentially with the pattern extractor
    pe_out = apply_pattern_extractor(encoder, fe_out)

    # Pass trough a last layer before sampling
    μ, logσ² = apply_latent_in(encoder, pe_out)

    return μ, logσ²
end

Flux.@functor Encoder

struct Decoder{M,LI,D,R}

    model_type::M

    latent_out::LI
    diffeq::D
    reconstructor::R

    @doc raw"""
        Decoder(model_type, decoder_layers)

    Construct a decoder of a LatentDiffEqModel.

    # Arguments
    model_type: GOKU() or LatentODE()
    decoder_layers: Could be specified manually or produced by default_layers
    """
    function Decoder(model_type, decoder_layers)
        M = typeof(model_type)
        LI, D, R = typeof.(decoder_layers)
        new{M,LI,D,R}(model_type, decoder_layers...)
    end
end

function (decoder::Decoder)(l̃, t)

    # Pass sampled latent states through a latent_out layer
    l̂ = apply_latent_out(decoder, l̃)

    # Integrate differential equations
    ẑ = diffeq_layer(decoder, l̂, t)

    # Apply reconstructor independently to each time frame
    x̂ = apply_reconstructor(decoder, ẑ)

    return x̂, ẑ, l̂
end

Flux.@functor Decoder