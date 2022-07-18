### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 47e5098e-0547-11ed-3bcb-b5efb72654e8
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ e6ac8d98-2e76-4cab-bc1d-770c98280d43
K = ProbSemiring{Float64}

# ╔═╡ 8ccaa6d2-9038-48a7-aace-1d9c6df7ff9c
full_fsm = FSM(
	[1 => one(K)],
	[
		(1, 1) => K(1/2),
		(1, 2) => K(1/3),
		(1, 3) => K(1/6),
		(2, 1) => K(1/2),
		(2, 2) => K(1/3),
		(2, 3) => K(1/6),
		(3, 1) => K(1/2),
		(3, 2) => K(1/3),
		(3, 3) => K(1/6),
	],
	[3 => one(K)],
	[Label("a"), Label("b"), Label("c")]
) |> renorm

# ╔═╡ b4615bba-e833-4b9b-b7fa-5c3bea7ad4b8
slr_T = MarkovModels.SparseLowRankMatrix(
	sparse([1], [2], K[3], 3, 3),
	reshape(K[1/2,  1/3, 1/6], 3, 1),
	ones(K, 3, 1)
) 

# ╔═╡ 12073c2d-8e66-4f34-a702-1ee6526487f6
fsm = FSM(
	full_fsm.α, 
	slr_T,
	full_fsm.ω,
	full_fsm.λ
) 

# ╔═╡ 72c617c9-e811-4730-ad49-7fc06d1dbcdd
fsm.T isa MarkovModels.SparseLowRankMatrix

# ╔═╡ 33dea991-1697-4a8f-8124-230da259e711
fsm.T̂.S[1, 2] .+ fsm.T̂.U[1, :] * fsm.T̂.V[2, :]'

# ╔═╡ cb5f43c9-86cd-4034-933c-c570ad20d494
fsm.T̂.U[1, :] * fsm.T̂.V[2, :]'

# ╔═╡ e18c245a-14ee-41f5-8871-d53afea3e9aa
fsm.T̂

# ╔═╡ 5fb91964-fde1-4ac8-ab91-ae8fd85633b2
full_fsm.T

# ╔═╡ e0abb5d1-ca47-4d42-b641-c6203b10059a
typeof(:)

# ╔═╡ ea0bd517-612f-456a-80fe-5ecca55e2245
reshape([1/2, 1/3, 1/6], 3, 1)

# ╔═╡ e659578e-3388-4d15-bc2c-3a65e794cea9
FSM(
	full_fsm.α,
	slr_T,
	full_fsm.ω,
	full_fsm.λ
)

# ╔═╡ c1551ff0-81d9-4d7f-839c-aea76504dcf0
ω = full_fsm.ω

# ╔═╡ 37ee4c36-e726-437f-bc67-2c5eb7e61e30
size(slr_T.U)

# ╔═╡ e0a1b661-6ade-46c2-bde1-619fc7ec0c80
vcat(slr_T, reshape(full_fsm.ω, 1, :))

# ╔═╡ 2600a085-ba43-4c3e-933f-42128772bf36
hcat(slr_T, full_fsm.ω)

# ╔═╡ 2757d5ea-18ad-4a89-84d5-d056085b83e8
vcat(slr_T, reshape(full_fsm.ω, 1, :))

# ╔═╡ 9fd4e2b1-3f55-49d1-8f3a-4e3c78e8a3fe
hcat(slr_T, full_fsm.ω)

# ╔═╡ d38defcc-2b6a-40fc-b972-2c97c3e0dd22
hcat(slr_T.V, full_fsm.ω)

# ╔═╡ f341686e-5108-4a63-906c-580bf0492a95
cat(slr_T.V, full_fsm.ω)

# ╔═╡ 4f6cb0bd-5418-4b14-b1a7-6b0ebc18e0e9
slr_T.V

# ╔═╡ c7368f4f-213b-41cd-bf91-362fd36a54aa
S2 = vcat(slr_T.S, reshape(full_fsm.ω, 1, :))

# ╔═╡ 4396d66a-d963-45f9-a79b-c386a4646548
U2 = vcat(slr_T.U, reshape(full_fsm.ω, 1, :)')

# ╔═╡ 6792930c-76f6-4e76-b9b7-e332097c374a
slr_T.U * ones(K, 4,1)'

# ╔═╡ 3c8f7335-1b45-45ad-ba2e-78b281d56c46
reshape(full_fsm.ω, 1, :)

# ╔═╡ b318dee3-b733-4957-93c4-5ee83cc93565
slr_T.U

# ╔═╡ Cell order:
# ╠═47e5098e-0547-11ed-3bcb-b5efb72654e8
# ╠═e6ac8d98-2e76-4cab-bc1d-770c98280d43
# ╠═8ccaa6d2-9038-48a7-aace-1d9c6df7ff9c
# ╠═b4615bba-e833-4b9b-b7fa-5c3bea7ad4b8
# ╠═12073c2d-8e66-4f34-a702-1ee6526487f6
# ╠═72c617c9-e811-4730-ad49-7fc06d1dbcdd
# ╠═33dea991-1697-4a8f-8124-230da259e711
# ╠═cb5f43c9-86cd-4034-933c-c570ad20d494
# ╠═e18c245a-14ee-41f5-8871-d53afea3e9aa
# ╠═5fb91964-fde1-4ac8-ab91-ae8fd85633b2
# ╠═e0abb5d1-ca47-4d42-b641-c6203b10059a
# ╠═ea0bd517-612f-456a-80fe-5ecca55e2245
# ╠═e659578e-3388-4d15-bc2c-3a65e794cea9
# ╠═c1551ff0-81d9-4d7f-839c-aea76504dcf0
# ╠═37ee4c36-e726-437f-bc67-2c5eb7e61e30
# ╠═e0a1b661-6ade-46c2-bde1-619fc7ec0c80
# ╠═2600a085-ba43-4c3e-933f-42128772bf36
# ╠═2757d5ea-18ad-4a89-84d5-d056085b83e8
# ╠═9fd4e2b1-3f55-49d1-8f3a-4e3c78e8a3fe
# ╠═d38defcc-2b6a-40fc-b972-2c97c3e0dd22
# ╠═f341686e-5108-4a63-906c-580bf0492a95
# ╠═4f6cb0bd-5418-4b14-b1a7-6b0ebc18e0e9
# ╠═c7368f4f-213b-41cd-bf91-362fd36a54aa
# ╠═4396d66a-d963-45f9-a79b-c386a4646548
# ╠═6792930c-76f6-4e76-b9b7-e332097c374a
# ╠═3c8f7335-1b45-45ad-ba2e-78b281d56c46
# ╠═b318dee3-b733-4957-93c4-5ee83cc93565
