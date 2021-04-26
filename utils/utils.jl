
################################################################################
## Loss definitions

function loss_batch(model::LatentDiffEqModel, λ, x, t, af)

    # Make prediction
    X̂, μ, logσ² = model(x, t)
    x̂, ẑ, ẑ₀, = X̂

    # Compute reconstruction (and differential) loss
    reconstruction_loss = vector_mse(x, x̂)

    # Compute KL losses from parameter and initial value estimation
    kl_loss = sum( [ mean(sum(kl.(μ[i], logσ²[i]), dims=1)) for i in 1:length(μ) ] )

    return reconstruction_loss + kl_loss
end

kl(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# make it better for gpu
# CUDA.@cufunc kl(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

function loss_batch(model::LatentDiffEqModel, discriminator, λ, x, t, af)

    # Make prediction
    X̂, μ, logσ² = model(x, t)
    x̂, ẑ, ẑ₀, = X̂

    # Discriminator loss
    d_x = discriminator.(x)
    d_x̂ = discriminator.(x̂)

    discriminator_loss = vector_mse(d_x, d_x̂)

    # Compute reconstruction (and differential) loss
    reconstruction_loss = vector_mse(x, x̂)

    # Compute KL losses from parameter and initial value estimation
    kl_loss = sum( [ mean(sum(kl.(μ[i], logσ²[i]), dims=1)) for i in 1:length(μ) ] )
    
    return reconstruction_loss + kl_loss + 0.003f0*discriminator_loss
end

function loss_batch(model::LatentDiffEqModel, discriminator_img, discriminator_seq, λ, x, t, af)

    # Make prediction
    X̂, μ, logσ² = model(x, t)
    x̂, ẑ, ẑ₀, = X̂

    # Discriminator for images
    Dimg_x = discriminator_img.(x)
    Dimg_x̂ = discriminator_img.(x̂)
    discriminator_img_loss = rec_loss(Dimg_x, Dimg_x̂)

    # Discriminator for images
    Dseq_x = discriminator_seq.(x)[end]
    reset!(discriminator_seq)
    Dseq_x̂ = discriminator_seq.(x̂)[end]
    reset!(discriminator_seq)

    discriminator_seq_loss = vector_mse(Dseq_x, Dseq_x̂)

    # Compute reconstruction (and differential) loss
    reconstruction_loss = vector_mse(x, x̂)

    # Compute KL losses from parameter and initial value estimation
    kl_loss = sum( [ mean(sum(kl.(μ[i], logσ²[i]), dims=1)) for i in 1:length(μ) ] )
    
    return reconstruction_loss + kl_loss + 0.003f0*discriminator_img_loss + 0.003f0*discriminator_seq_loss
end

function vector_mse(x, x̂)
    res = zero(eltype(x[1]))
    for i in eachindex(x)
        res += mean((x[i] .- x̂[i]).^2)
    end
    res /= length(x)
end


## annealing factor parameters
# start_af: start value of annealing factor
# end_af: end value of annealing factor
# ae: annealing epochs - after these epochs, the annealing factor is set to the end value.
# epoch: current epoch
# mb_id: current number of mini batch
# mb_amount: amount of mini batches
function annealing_factor(start_af, end_af, ae, epoch, mb_id, mb_amount)

    if ae > 0
        if epoch < ae
            return Float32(start_af + (end_af - start_af) * (mb_id + epoch * mb_amount) / (ae * mb_amount))
        else
            return Float32(end_af)
        end
    else
        return 1.0f0
    end

end


################################################################################
## Data pre-processing

function normalize_to_unit_segment(X)
    min_val = minimum(X)
    max_val = maximum(X)

    X̂ = (X .- min_val) ./ (max_val - min_val)
    return X̂, min_val, max_val
end

denormalize_unit_segment(X̂, min_val, max_val) = X̂ .* (max_val .- min_val) .+ min_val


################################################################################
## Training help function

function time_loader(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))

    for i in 1:size(x,3)
        x_[:,:,i] = x[:,rand_time(full_seq_len, seq_len),i]
        # x_[:,:,i] = x[:,1:seq_len,i]
    end
    return Flux.unstack(x_, 2)
end

function rand_time(full_seq_len, seq_len)
    start_time = rand(1:full_seq_len - seq_len)
    idxs = start_time:start_time+seq_len-1
    return idxs
end