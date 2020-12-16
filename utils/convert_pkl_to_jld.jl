using PyCall
using Conda
using JLD2, FileIO
torch = pyimport("torch")

dataset = "pendulum_friction-less"

root_dir = @__DIR__
A = readdir("$root_dir/../$dataset/data/pkl")
filter!(e->e≠".DS_Store",A)
jld_save_path = "$root_dir/../$dataset/data/jld"
mkpath(jld_save_path)

for data in A
    data_loaded = torch.load("$root_dir/../$dataset/data/pkl/$data")
    name = data[1:end-4]
    save("$jld_save_path/$name.jld2", name, data_loaded)
end

