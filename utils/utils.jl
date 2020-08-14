
################################################################################
## Loss definitions

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# make it better for gpu
CUDA.@cufunc KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
# KL(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ


function rec_loss(x, pred_x)

    # Data prep
    pred_x_stacked = Flux.stack(pred_x, 3)
    x_stacked = Flux.stack(x, 3)

    # Residual loss
    res = pred_x_stacked - x_stacked
    res_average = sum(mean((res).^2, dims = (2, 3)))

    # Differential residual loss
    res_diff = diff(pred_x_stacked, dims = 3) - diff(x_stacked, dims = 3)
    res_diff_average = sum(mean((res_diff).^2, dims = (2, 3)))

    return res_average + 1000*res_diff_average
end

function rec_loss_mahalanobis(x, pred_x)

    # Data prep
    pred_x_stacked = permutedims(Flux.stack(pred_x, 3), [1,3,2])
    x_stacked = permutedims(Flux.stack(x, 3), [1,3,2])
    pred_x_stacked_diff = permutedims(diff(pred_x_stacked, dims = 3), [1,3,2])
    x_stacked_diff = permutedims(diff(x_stacked, dims = 3), [1,3,2])
    # now the size is (minibatch_size, num_vars, seq_len)

    pred_x_stacked = reshape(pred_x_stacked, (size(pred_x_stacked)[1], size(pred_x_stacked)[2]*size(pred_x_stacked)[3]))
    x_stacked = reshape(x_stacked, (size(x_stacked)[1], size(x_stacked)[2]*size(x_stacked)[3]))
    pred_x_stacked_diff = reshape(pred_x_stacked_diff, (size(pred_x_stacked_diff)[1], size(pred_x_stacked_diff)[2]*size(pred_x_stacked_diff)[3]))
    x_stacked_diff = reshape(x_stacked_diff, (size(x_stacked_diff)[1], size(x_stacked_diff)[2]*size(x_stacked_diff)[3]))
    # now the size is (num_vars, seq_len*minibatch_size)

    ss = DistributionsAD.suffstats(MvNormal, convert(Array{Float64}, x_stacked)) #conversion because DistributionsAD only works with Float64
    d = DistributionsAD.fit_mle(MvNormal, ss)
    cov_mat = d.C
    Q = deepcopy(inv(cov_mat))
    print(Q)
    # dist_array = colwise(Mahalanobis(Q), pred_x_stacked, x_stacked) #can't differentiate loopinfo
    # dist = sum(dist_array)
    dist = 0
    for i in range(1,size(x_stacked)[2]; step=1)
        dist += mahalanobis(pred_x_stacked[:,i], x_stacked[:,i], Q)
    end
    print(dist)

    ss_diff = DistributionsAD.suffstats(MvNormal, convert(Array{Float64}, x_stacked_diff)) #conversion because DistributionsAD only works with Float64
    d_diff = DistributionsAD.fit_mle(MvNormal, ss_diff)
    cov_mat_diff = d_diff.C
    Q_diff = deepcopy(inv(cov_mat_diff))
    print(Q_diff)

    # dist_diff_array = colwise(Mahalanobis(Q), pred_x_stacked_diff, x_stacked_diff) #can't differentiate loopinfo
    # dist_diff = sum(dist_diff_array)
    dist_diff = 0
    for i in range(1,size(x_stacked_diff)[2]; step=1)
        dist_diff += mahalanobis(pred_x_stacked_diff[:,i], x_stacked_diff[:,i], Q)
    end
    print(dist_diff)
    return dist + 1000*dist_diff
end

function loss_batch(model::AbstractModel, λ, x, t, af, rec_loss_type="L2")

    # Make prediction
    lat_var, pred_x, pred = model(x, t)

    # Compute reconstruction (and differential) loss
    if rec_loss_type == "mahalanobis"
        reconstruction_loss = rec_loss_mahalanobis(x, pred_x)
    else
        reconstruction_loss = rec_loss(x, pred_x)
    end


    # Compute KL losses from parameter and initial value estimation
    # kl_loss = 0
    # for i in 1:length(lat_var)
    #     μ, logσ² = lat_var[i]
    #     kl_loss += mean(sum(KL.(μ, logσ²), dims=1))
    # end

    # Filthy one liner that does the for loop above # lit
    kl_loss = sum( [ mean(sum(KL.(lat_var[i][1], lat_var[i][1]), dims=1)) for i in 1:length(lat_var) ] )

    return reconstruction_loss + af*(kl_loss)
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

# normalize raw data passed as a 3D array (input_dim, time, trajectories)
function normalize_Z(data)

      data = Flux.unstack(data, 3)

      μ = 0.
      σ = 0.
      for i in 1:size(data,1)
            for j in 1:size(data[1],1)
                  μ = mean(data[i][j,:])
                  σ = std(data[i][j,:])
                  data[i][j,:] = ( data[i][j,:] .- μ ) ./ σ
            end
      end

      data = Flux.stack(data, 3)

      return norm_data

end


################################################################################
## Training help function

function rand_time(full_seq_len, seq_len)
    start_time = rand(1:full_seq_len - seq_len)
    idxs = start_time:start_time+seq_len-1
    return idxs
end

function time_loader(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))

    for i in 1:size(x,3)
        x_[:,:,i] = x[:,rand_time(full_seq_len, seq_len),i]
    end

    return Flux.unstack(x_, 2)

end
