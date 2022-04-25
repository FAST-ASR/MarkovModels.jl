### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 6e84e042-b5a5-11ec-3bda-e7f5e804b97e
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using MarkovModels
	using Semirings
	using SparseArrays
end

# ╔═╡ 341f660d-d1be-4fa7-9d11-763da2181941
Label(x) = UnionConcatSemiring(Set([SymbolSequence([x])]))

# ╔═╡ 238148e8-dfd8-454f-abb7-9571d09958a6
K = ProbSemiring{Float32}

# ╔═╡ eb27ffd7-2496-4edb-a2b0-e5828278410c
fsm1 = FSM{K}(
	[1, 0],
	[0.5 0.5;
	  0.0 0.5],
	[0, 0.5],
	[Label(:1), Label(:2)]
) |> renorm

# ╔═╡ 140ef71a-d95e-4d2b-ad50-7a42c3125d7f
fsm2 = FSM{K}(
	[1, 0, 0],
	[0.5 0.5 0.0;
	  0.0 0.5 0.5;
	  0.0 0.0 0.5;],
	[0, 0, 0.5],
	[Label(:1), Label(:2), Label(:3)]
)

# ╔═╡ 521d5474-ae67-48ef-8dab-73334fbe5b6a
renorm(fsm2)

# ╔═╡ 56faf31f-32a3-4ef2-a094-06f5a2987ab6
union(fsm1, fsm2)

# ╔═╡ 813026d5-6171-4749-b6e2-b38e67a78e9f
concat(fsm1, fsm2)

# ╔═╡ 9e09acd6-780c-44f2-866a-070b10a46c7b
fsm = FSM{K}(
	[2, 1],
	[0 1;
	 1 0],
	[1, 0],
	[Label(:a), Label(:b)]
) |> renorm

# ╔═╡ c29c020d-829d-4f5f-804b-f0d6591329fc
fsm ∘ [fsm1 |> renorm, fsm2 |> renorm]

# ╔═╡ 4c7d472e-c69b-4757-9e6c-5444e6ca5751
[fsm1.T zero(fsm2.T); zero(fsm1.T) fsm2.T]

# ╔═╡ c2c25125-cc7f-423d-9ce1-b48e06b477ae


# ╔═╡ 27f2175f-985f-419d-819f-58e101554d2d
g[:, 1] .+  sparsevec(t) 

# ╔═╡ 0d8f1915-d2fc-456d-acc1-006ded794478
sparsevec(l)

# ╔═╡ 8ae7342a-a7a9-4b65-aad7-91e641b5cfef
M * sparse(l[:, :])

# ╔═╡ fedc5764-3115-4004-b71d-a17279aa7b4c
sprandn(10, 10, 0.1) * randn(10)

# ╔═╡ 46f419a6-d320-4bd6-a26e-f78aa9866f3e
one(UnionConcatSemiring)  + fsm1.λ[1]

# ╔═╡ 70ad22a7-6c95-4079-859b-9eacf77e9a91
2*0.166 + 0.666

# ╔═╡ 2afa17db-fbff-45b2-b2a9-c4c2ac907121
D = vcat(hcat(fsm.T, fsm.ω), zeros(K, 1, length(fsm.ω)+1))

# ╔═╡ 18847048-c67c-4058-a004-210b6ac48dd2
D[end,end] = 1

# ╔═╡ dc98646a-2d8b-419c-b639-f205ee012c0c
D

# ╔═╡ 841fc7d7-48d4-4c44-b322-cdf486717770
one(UnionConcatSemiring)

# ╔═╡ edfaeb45-37fd-4dd0-94cf-1fd7fdf49260
zero(UnionConcatSemiring)

# ╔═╡ 96911c39-f871-41e8-b83d-05f4633ce58e
A11 = A12 = A21 = A22 = rand(4, 4);

# ╔═╡ 49a51adb-f41a-42d5-8dec-363d19a96825
[A11  zero(A11);
 zero(A11) A22]

# ╔═╡ 87223769-e9ef-4982-bc4a-d90d63931df2
zero(A11)

# ╔═╡ 78d44e1d-ab26-4ccd-b518-401b6a3bbf5a
y = sparse(1:3, ones(3), 2*ones(3))

# ╔═╡ b04a2028-9233-4016-b834-3895df3b7a9e
blockdiag(y, y, y)

# ╔═╡ 88a72285-8521-4e5b-9b5c-f76b6ccd207f
v = vcat(fsm.α, zero(K))

# ╔═╡ b0fff691-e5dc-4947-8571-de66e791026f
D' * (D' * (D' * (D' * (D' * (D' * (D' * v))))))

# ╔═╡ 657e4548-c776-4f23-a1b6-128d898fc155
prod(fsm.λ) .+ fsm.λ

# ╔═╡ 90c714f7-9389-4c45-a4a9-8cea264e9b40
(fsm.λ[1] * fsm.λ[2] + fsm.λ[3])

# ╔═╡ 4023155a-3b3d-4fb8-9b62-5a4772d7306b
fsm.α[3] == zero(K)

# ╔═╡ f883eddd-3469-49ba-8614-6768f11492f5
zero(fsm.α[3])

# ╔═╡ 546f7484-c945-4fa7-bc07-1099bf7b7591
x = ProbSemiring[1.0, 0.0, 0]

# ╔═╡ e9b896e4-46ba-484f-8bff-0c463228b77a
UnionConcatSemiring(Set([SymbolSequence([:a, :b, :c])]))

# ╔═╡ 0fbe3317-3d24-41fc-84ca-028766731083
nonzeros

# ╔═╡ ad3b9922-601c-4ec3-81d2-f91df8b7c42c
join([1, 2, 3])

# ╔═╡ Cell order:
# ╠═6e84e042-b5a5-11ec-3bda-e7f5e804b97e
# ╠═341f660d-d1be-4fa7-9d11-763da2181941
# ╠═238148e8-dfd8-454f-abb7-9571d09958a6
# ╠═eb27ffd7-2496-4edb-a2b0-e5828278410c
# ╠═140ef71a-d95e-4d2b-ad50-7a42c3125d7f
# ╠═521d5474-ae67-48ef-8dab-73334fbe5b6a
# ╠═56faf31f-32a3-4ef2-a094-06f5a2987ab6
# ╠═813026d5-6171-4749-b6e2-b38e67a78e9f
# ╠═9e09acd6-780c-44f2-866a-070b10a46c7b
# ╠═c29c020d-829d-4f5f-804b-f0d6591329fc
# ╠═4c7d472e-c69b-4757-9e6c-5444e6ca5751
# ╠═c2c25125-cc7f-423d-9ce1-b48e06b477ae
# ╠═27f2175f-985f-419d-819f-58e101554d2d
# ╠═0d8f1915-d2fc-456d-acc1-006ded794478
# ╠═8ae7342a-a7a9-4b65-aad7-91e641b5cfef
# ╠═fedc5764-3115-4004-b71d-a17279aa7b4c
# ╠═46f419a6-d320-4bd6-a26e-f78aa9866f3e
# ╠═70ad22a7-6c95-4079-859b-9eacf77e9a91
# ╠═2afa17db-fbff-45b2-b2a9-c4c2ac907121
# ╠═18847048-c67c-4058-a004-210b6ac48dd2
# ╠═dc98646a-2d8b-419c-b639-f205ee012c0c
# ╠═841fc7d7-48d4-4c44-b322-cdf486717770
# ╠═edfaeb45-37fd-4dd0-94cf-1fd7fdf49260
# ╠═96911c39-f871-41e8-b83d-05f4633ce58e
# ╠═49a51adb-f41a-42d5-8dec-363d19a96825
# ╠═87223769-e9ef-4982-bc4a-d90d63931df2
# ╠═78d44e1d-ab26-4ccd-b518-401b6a3bbf5a
# ╠═b04a2028-9233-4016-b834-3895df3b7a9e
# ╠═88a72285-8521-4e5b-9b5c-f76b6ccd207f
# ╠═b0fff691-e5dc-4947-8571-de66e791026f
# ╠═657e4548-c776-4f23-a1b6-128d898fc155
# ╠═90c714f7-9389-4c45-a4a9-8cea264e9b40
# ╠═4023155a-3b3d-4fb8-9b62-5a4772d7306b
# ╠═f883eddd-3469-49ba-8614-6768f11492f5
# ╠═546f7484-c945-4fa7-bc07-1099bf7b7591
# ╠═e9b896e4-46ba-484f-8bff-0c463228b77a
# ╠═0fbe3317-3d24-41fc-84ca-028766731083
# ╠═ad3b9922-601c-4ec3-81d2-f91df8b7c42c
