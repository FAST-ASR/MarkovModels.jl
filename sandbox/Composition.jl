### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ aba927a2-1032-11ed-037f-452f342c54c6
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ 580e697d-8d0d-464c-b73d-5e00228a7317
K = ProbSemiring{Float64}

# ╔═╡ 8ce8ec21-13e0-4bb6-a272-d8778106cc2c
A = SparseFSA(
	[1 => one(K)],
	[(1, 2) => one(K) + one(K), (2, 3) => one(K), (2, 1) => one(K),
	 (1, 3) => one(K)],
	[2 => one(K), 3 => one(K)],
	["a", "b", "c"]
)

# ╔═╡ 6b07528a-40f1-4021-8606-ced802a30ed9
B = SparseFSA(
	[1 => one(K)],
	[(1, 2) => K(3), (2, 3) => one(K), (3, 4) => one(K), (2, 1) => K(2)],
	[4 => one(K)],
	["a", "b", "a", "c"]
)

# ╔═╡ 747fb8d4-e571-496c-9d95-97963b9be505
a = Set([1, 2, 3])

# ╔═╡ dd48808b-14f0-4c7a-b295-2a87f774bbfc
push!(a, [1, 2])

# ╔═╡ 4c74b9e7-a4cf-485b-9ade-a922ecc2a399
begin
	local visited = Set()
	browse(A) do v
		@show v
		I, V = findnz(v)
		Iz, Vz = [], eltype(V)[]
		for (i, v) in zip(I, V)
			if i ∉ visited
				push!(Iz, i)
				push!(Vz, v)
				push!(visited, i)
			end
		end
		sparsevec(Iz, Vz, length(v))
	end
end

# ╔═╡ 777fbcdd-db46-48a9-8d8a-09e7fe3ef47a
FSA(
	kron(A.α, B.α),
	kron(A.T, B.T),
	kron(A.ω, B.ω),
	kron(A.λ, B.λ)
)

# ╔═╡ e609d8b8-f933-4b51-b5f3-f79a75cab668
I, J, V = findnz(kron(A.T, B.T))

# ╔═╡ ee2e5ea8-1739-4fae-992f-279096b94da2
begin 
	Iz, Jz, Vz = [], [], K[]
	for (i, j, v) in zip(I, J, V)
		ia, ib = (i - 1) ÷ nstates(B) + 1, (i - 1) % nstates(B) + 1 
		ja, jb = (j - 1) ÷ nstates(B) + 1, (j - 1) % nstates(B) + 1 
		
		if A.λ[ia] == B.λ[ib] && A.λ[ja] == B.λ[jb]
			push!(Iz, i)
			push!(Jz, j)
			push!(Vz, v)
		end
	end
	
	FSA(
		kron(A.α, B.α),
		sparse(Iz, Jz, Vz),
		kron(A.ω, B.ω),
		kron(A.λ, B.λ)
	)
end

# ╔═╡ 6d6c71bc-3eeb-4803-9815-89e1a2d8b4c0
11 ÷ 4

# ╔═╡ 3a76f6a1-d878-417f-b9aa-eea50c5f79ee
2 ÷ 2 + 1

# ╔═╡ f1da6e7d-0599-4bb6-b32b-447ca44b55b7
kron(A.λ, B.λ)

# ╔═╡ 9377abe8-4d0e-42ca-abbb-9f500342d32a
function forward(μ::Function, fsa_a::FSA, fsa_b::FSA)
	vₙ = fsa_a.α
	uₙ = fsa_b.α
	retval = [μ(vₙ, uₙ)]
	
	#while nnz(retval[end]) > 0
	for i in 1:4
		vₙ = fsa_a.T' * vₙ
		uₙ = fsa_b.T' * uₙ
		push!(retval, μ(vₙ, uₙ))
	end
	retval[1:end-1]
end

# ╔═╡ 5b04bfa6-18d0-4510-9a6f-81fce73c9197
function backward(μ::Function, fsa_a::FSA, fsa_b::FSA)
	vₙ = fsa_a.ω
	uₙ = fsa_b.ω
	retval = [μ(vₙ, uₙ)]
	
	#while nnz(retval[begin]) > 0
	for i in 1:4
		vₙ = fsa_a.T * vₙ
		uₙ = fsa_b.T * uₙ
		insert!(retval, 1, μ(vₙ, uₙ))
	end
	retval[2:end]
end

# ╔═╡ 3c113fec-f53e-4901-83cf-0396d68125cb
begin
	local fwd_visited = Set()
	fwd = forward(A, B) do vₙ, uₙ
		Iv, Vv = findnz(vₙ)
		Iu, Vu = findnz(uₙ)
		Iw, Vw = [], eltype(vₙ)[]
		for iv = 1:length(Iv), iu = 1:length(Iu)
			sv, su = Iv[iv], Iu[iu]
			if A.λ[sv] == B.λ[su] && (sv, su) ∉ fwd_visited 
				push!(fwd_visited, (sv, su))
				push!(Iw, sv)
				push!(Vw, Vv[iv] * Vu[iu])
			end
		end
		sparsevec(Iw, Vw, length(vₙ))
	end

	local bwd_visited = Set()
	bwd = backward(A, B) do vₙ, uₙ
		Iv, Vv = findnz(vₙ)
		Iu, Vu = findnz(uₙ)
		Iw, Vw = [], eltype(vₙ)[]
		for iv = 1:length(Iv), iu = 1:length(Iu)
			sv, su = Iv[iv], Iu[iu]
			if A.λ[sv] == B.λ[su] && (sv, su) ∉ bwd_visited 
				push!(bwd_visited, (sv, su))
				push!(Iw, sv)
				push!(Vw, Vv[iv] * Vu[iu])
			end
		end
		sparsevec(Iw, Vw, length(vₙ))
	end
	

	local T = zero(A.T)
	for n = 1:length(fwd)-1
		T += spdiagm(fwd[n]) * A.T * spdiagm(bwd[n+1])
	end
	sum(T)
	
	FSA(
		fwd[1],
		T,
		bwd[end],
		A.λ
	)
end

# ╔═╡ 6c8edb52-bcb3-40f0-92b0-f688d18c231a
spdiagm(fwd[1]) * A.T * spdiagm(bwd[3])

# ╔═╡ b69d3e9a-ec23-44da-8c5a-d6508f536854
bwd[2]

# ╔═╡ 65c2b0af-4c65-47c0-adcf-58f0aa05e19e
bwd[end]

# ╔═╡ 9ff6e379-cef4-4c97-86e8-6a5ae5de4858
FSA(
	fwd[1],
	spdiagm(fwd[1]) * A.T * spdiagm(bwd[2]) + 
	spdiagm(fwd[2]) * A.T * spdiagm(bwd[3]),
	bwd[end],
	A.λ
)

# ╔═╡ Cell order:
# ╠═aba927a2-1032-11ed-037f-452f342c54c6
# ╠═580e697d-8d0d-464c-b73d-5e00228a7317
# ╠═8ce8ec21-13e0-4bb6-a272-d8778106cc2c
# ╠═6b07528a-40f1-4021-8606-ced802a30ed9
# ╠═747fb8d4-e571-496c-9d95-97963b9be505
# ╠═dd48808b-14f0-4c7a-b295-2a87f774bbfc
# ╠═4c74b9e7-a4cf-485b-9ade-a922ecc2a399
# ╠═777fbcdd-db46-48a9-8d8a-09e7fe3ef47a
# ╠═e609d8b8-f933-4b51-b5f3-f79a75cab668
# ╠═ee2e5ea8-1739-4fae-992f-279096b94da2
# ╠═6d6c71bc-3eeb-4803-9815-89e1a2d8b4c0
# ╠═3a76f6a1-d878-417f-b9aa-eea50c5f79ee
# ╠═f1da6e7d-0599-4bb6-b32b-447ca44b55b7
# ╠═9377abe8-4d0e-42ca-abbb-9f500342d32a
# ╠═5b04bfa6-18d0-4510-9a6f-81fce73c9197
# ╠═3c113fec-f53e-4901-83cf-0396d68125cb
# ╠═6c8edb52-bcb3-40f0-92b0-f688d18c231a
# ╠═b69d3e9a-ec23-44da-8c5a-d6508f536854
# ╠═65c2b0af-4c65-47c0-adcf-58f0aa05e19e
# ╠═9ff6e379-cef4-4c97-86e8-6a5ae5de4858
