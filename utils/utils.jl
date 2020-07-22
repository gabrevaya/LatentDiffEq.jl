

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0
#the following works better for gpu
# CuArrays.@cufunc KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0
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
            return start_af + (end_af - start_af) * (Float32(mb_id + epoch * mb_amount) / Float32(ae * mb_amount))
        else
            return end_af
        end
    else
        return 1.0
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
#
# # overload data loader function so that it picks random start times for each sample, of size seq_len
# function time_idxs(seq_len, time_len)
#     start_time = rand(1:time_len - seq_len)
#     idxs = start_time:start_time+seq_len-1
# end
#
# function Flux.Data._getobs(data::AbstractArray, i)
#     features, time_len, obs = size(data)
#     seq_len::Int
#     data_ = Array{Float32, 3}(undef, (features, seq_len, length(i)))
#
#     for (idx, batch_idx) in enumerate(i)
#         data_[:,:, idx] =
#         data[:, time_idxs(seq_len, time_len), batch_idx]
#     end
#     return Flux.unstack(data_, 2)
# end
