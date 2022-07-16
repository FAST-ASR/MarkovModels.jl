### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 3b586fda-ebc5-11ec-3816-e54f891871f0
begin
	using Pkg
	Pkg.activate("../")

	using Adapt
	using CUDA
	using CUDA.CUSPARSE
	using LinearAlgebra
	using Plots
	using Semirings
	using Serialization
	using SparseArrays
	using BenchmarkTools
	using JSON

	using Revise
	using MarkovModels
end

# ╔═╡ 5114ca9c-e03a-4d80-85a8-868d8b398eb8
K = LogSemiring{Float32}

# ╔═╡ a72ed4d4-7dfb-46ee-8339-96b0bbace635
rootdir = "/home/ubuntu/Exps/lfmmi2"

# ╔═╡ 902386b6-d7ca-4d67-8b5a-466e6d8f8a89
dataset = joinpath(rootdir, "lfmmi-dataset-c023e66a01c5ae2f9d629da658914f344b9c4a719d612f55441c57350d29bb77")

# ╔═╡ 6c31e9dd-f121-4d2e-9d86-0976529ad49b
Ta = CuArray

# ╔═╡ 26bd2857-5889-4788-aa51-87225e3e5226
numpdf = 96

# ╔═╡ df01f038-5732-4814-a968-2c8e9dd524ba
fsmfile = joinpath(rootdir, "sup_s03_lm3/numfsms/train/fsm.scp")

# ╔═╡ b6662f38-16f2-4c72-baad-80102e1ac0ee
smapfile = joinpath(rootdir, "sup_s03_lm3/numfsms/train/smap.scp")

# ╔═╡ 4728d391-9a80-4e16-aaac-a57438b077d7
(3.719937562942505 * 100) / 3

# ╔═╡ a0d53e7f-0186-4148-b4eb-1f3ff1340c8e
gpu() = false

# ╔═╡ d63c9c25-a998-4401-ba93-adf24ff43c37
batchsize = 4

# ╔═╡ 7c981e9e-8598-48a0-9aeb-ae3fd3bf4b4e
begin
	fsms = []
	uttids = []
	seqlengths = []
	open(fsmfile, "r") do f
		for i in 1:batchsize
			line = readline(f)
			uttid, path = split(line)
			duration = JSON.parsefile(
				joinpath(rootdir, dataset, "train",
				 string(i ÷ 100, pad=2), string(100 * (i ÷ 100) + ((i-1) % 100), pad=4) * ".json")
			)["duration"]
			dur = Int(ceil(100 * duration / 3))
			push!(seqlengths, dur)
			push!(uttids, uttid)
			fsmpath = joinpath(rootdir, path)
			push!(fsms, deserialize(fsmpath))
		end
	end

	batch_fsms = gpu() ? adapt(Ta, rawunion(fsms...)) : rawunion(fsms...)
end;

# ╔═╡ cf089ec8-6a09-4405-98f3-112b00e4115d
adapt(CuArray, fsms[1]);

# ╔═╡ 401f1e23-31dd-44e7-8b87-6d67f411a8f6
adapt(CuArray, fsms[1].T̂)

# ╔═╡ cbe78279-c823-4a0b-a6a1-3151b270eb74
seqlengths

# ╔═╡ fbc4473e-08e8-4ec9-aba1-8ea996f7575a
begin
	Cs = []
	open(smapfile, "r") do f
		for i in 1:batchsize
			line = readline(f)
			uttid, path = split(line)
			smappath = joinpath(rootdir, path)
			push!(Cs, deserialize(smappath))
		end
	end
	Cs = gpu() ? CuSparseMatrixCSR.(adapt.(Ta, Cs)) : Cs
end;

# ╔═╡ a5cb6e28-1a69-4d1d-ab39-6fd7eb645061
begin
	denfsm = deserialize(joinpath(rootdir, "sup_s03_lm3/denominator.fsm"))
	denĈ = deserialize(joinpath(rootdir, "sup_s03_lm3/denominator.smap"))

	denfsm = gpu() ? adapt(Ta, denfsm) : denfsm
	denĈ = gpu() ? CuSparseMatrixCSR(adapt(Ta, denĈ)) : denĈ

	batch_denfsm = rawunion(repeat([denfsm], batchsize)...);
	batch_denĈ = repeat([denĈ], batchsize)
end;

# ╔═╡ 2e1bf652-2b02-45e6-8ced-5d3a038bdbf9
begin
	_pytorch_v = randn(batchsize, max(seqlengths...), numpdf)
	pytorch_v = gpu() ? adapt(Ta, _pytorch_v) : _pytorch_v
	v = permutedims(pytorch_v, (1, 3, 2))
end

# ╔═╡ dae82fef-b62e-4220-85ce-46b8c18403a1
L = map(t -> MarkovModels.expand(t...),
		 zip(eachslice(v, dims = 1), seqlengths))

# ╔═╡ 11d4cdce-55e9-4b6d-ba3d-f037b684cb00
a, ttl = MarkovModels.pdfposteriors(batch_fsms, L, Cs)

# ╔═╡ 752059cd-b7f6-4df0-a6b9-a7cbc99d687a
ttl

# ╔═╡ 5b7cc6df-e784-4a5b-b985-1893c5e1f4f4
# ╠═╡ disabled = true
#=╠═╡
@benchmark MarkovModels.pdfposteriors(batch_denfsm, L, batch_denĈ)
  ╠═╡ =#

# ╔═╡ ab1e01c3-a413-4071-a3cb-3abee7a52740
@time pZ, ttl_num = MarkovModels.pdfposteriors(batch_fsms, L, Cs)

# ╔═╡ 49d92be7-b03b-4bae-8d1f-23a8b8414cce
@time pZd, ttl_den = MarkovModels.pdfposteriors(batch_denfsm, L, batch_denĈ)

# ╔═╡ 7df99757-a041-4427-926d-8f0baf4c272b
heatmap(Array(pZ)[1, :, :])

# ╔═╡ cd9ffc16-1e18-4b12-886f-3f7ca2a65060
heatmap(Array(pZd)[1, :, :])

# ╔═╡ cd26c22e-40fd-4ebb-8d59-45044c192ddc
heatmap(Array(pZ - pZd)[1, :, :])

# ╔═╡ a8e756f8-76a9-48a8-b6e5-359084b4ee2c
# ╠═╡ disabled = true
#=╠═╡
@time MarkovModels.pdfposteriors(batch_fsms, L, Cs)
  ╠═╡ =#

# ╔═╡ 3ead9e04-e85d-489b-90ac-85280d720b5e
# ╠═╡ disabled = true
#=╠═╡
@benchmark MarkovModels.pdfposteriors(batchfsms, L, batchC)
  ╠═╡ =#

# ╔═╡ 7839af84-a6da-4453-9d82-90e617553571
# ╠═╡ disabled = true
#=╠═╡
A = αrecursion(
	batchfsms.α̂,
	batchfsms.T̂',
	batchC * ev,
)
  ╠═╡ =#

# ╔═╡ 2f10a219-0927-44fd-8a90-7e5589de1d3d
# ╠═╡ disabled = true
#=╠═╡
@benchmark αrecursion(
	batchfsms.α̂,
	batchfsms.T̂',
	batchC * ev,
)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═3b586fda-ebc5-11ec-3816-e54f891871f0
# ╠═5114ca9c-e03a-4d80-85a8-868d8b398eb8
# ╠═a72ed4d4-7dfb-46ee-8339-96b0bbace635
# ╠═902386b6-d7ca-4d67-8b5a-466e6d8f8a89
# ╠═6c31e9dd-f121-4d2e-9d86-0976529ad49b
# ╠═26bd2857-5889-4788-aa51-87225e3e5226
# ╠═df01f038-5732-4814-a968-2c8e9dd524ba
# ╠═b6662f38-16f2-4c72-baad-80102e1ac0ee
# ╠═7c981e9e-8598-48a0-9aeb-ae3fd3bf4b4e
# ╠═cf089ec8-6a09-4405-98f3-112b00e4115d
# ╠═401f1e23-31dd-44e7-8b87-6d67f411a8f6
# ╠═4728d391-9a80-4e16-aaac-a57438b077d7
# ╠═cbe78279-c823-4a0b-a6a1-3151b270eb74
# ╠═fbc4473e-08e8-4ec9-aba1-8ea996f7575a
# ╠═a5cb6e28-1a69-4d1d-ab39-6fd7eb645061
# ╠═2e1bf652-2b02-45e6-8ced-5d3a038bdbf9
# ╠═dae82fef-b62e-4220-85ce-46b8c18403a1
# ╠═11d4cdce-55e9-4b6d-ba3d-f037b684cb00
# ╠═752059cd-b7f6-4df0-a6b9-a7cbc99d687a
# ╠═5b7cc6df-e784-4a5b-b985-1893c5e1f4f4
# ╠═ab1e01c3-a413-4071-a3cb-3abee7a52740
# ╠═49d92be7-b03b-4bae-8d1f-23a8b8414cce
# ╠═a0d53e7f-0186-4148-b4eb-1f3ff1340c8e
# ╠═d63c9c25-a998-4401-ba93-adf24ff43c37
# ╠═7df99757-a041-4427-926d-8f0baf4c272b
# ╠═cd9ffc16-1e18-4b12-886f-3f7ca2a65060
# ╠═cd26c22e-40fd-4ebb-8d59-45044c192ddc
# ╠═a8e756f8-76a9-48a8-b6e5-359084b4ee2c
# ╠═3ead9e04-e85d-489b-90ac-85280d720b5e
# ╠═7839af84-a6da-4453-9d82-90e617553571
# ╠═2f10a219-0927-44fd-8a90-7e5589de1d3d
