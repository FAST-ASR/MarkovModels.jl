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

# ╔═╡ 093fda8f-4ce7-4cc2-90ee-1bef16a7c915
md"""
```math
\begin{align}
	\mathbf{v}_n = \mathbf{T}^\top \mathbf{v}_{n-1}
\end{align}
```
"""

# ╔═╡ 9124ca89-b612-48ab-ba89-1a671265575e


# ╔═╡ 597e70a6-9fe7-4168-be1a-cdf17ce67c40
md"""
```math
\begin{align}
	\mathbf{T}_{GL} = \begin{bmatrix}
		\mathbf{I}_{w1} \otimes \mathbf{T}_{w1} &  & \\
		& \mathbf{I}_{w1} \otimes \mathbf{T}_{w2} & \\
		& & \ddots
	\end{bmatrix}
		+ \begin{bmatrix}
			\mathbf{I}_{w1} \otimes \boldsymbol{\alpha}_{w1} &  & \\
			& \mathbf{I}_{w1} \otimes \boldsymbol{\alpha}_{w2} & \\
			& & \ddots
		\end{bmatrix} \mathbf{T}_{G} 
	\begin{bmatrix}
		\mathbf{I}_{w1} \otimes \boldsymbol{\omega}_{w1} &  & \\
		& \mathbf{I}_{w1} \otimes \boldsymbol{\omega}_{w2} & \\
		& & \ddots
	\end{bmatrix}^\top
\end{align}
```
"""

# ╔═╡ ba70bef7-161e-4060-9716-3ed3f8c4616d


# ╔═╡ 4ff95290-2872-42a2-8725-e32e730e2fc9
md"""
```math
\begin{align}
	\begin{bmatrix}
		\mathbf{I}_A \otimes \mathbf{T}_A &  & \\
		& \mathbf{I}_B \otimes \mathbf{T}_{B} & \\
		& & \ddots
	\end{bmatrix} 
	\begin{bmatrix}
		\mathbf{1}_A \otimes \mathbf{v}_{A} \\
		\mathbf{1}_B \otimes \mathbf{v}_{B} \\
		\vdots
	\end{bmatrix} = 
	\begin{bmatrix}
		\mathbf{1}_A \otimes (\mathbf{T}_A \mathbf{v}_{A}) \\
		\mathbf{1}_B \otimes (\mathbf{T}_B \mathbf{v}_{B}) \\
		\vdots
	\end{bmatrix}
\end{align}
```
"""

# ╔═╡ 09aaf348-fc20-4985-bb7d-c25d04659a8a
md"""
```
G = ... # Language model 
L = ... # Lexicon
H = ... # HMMs
LG = k2.compose(L, G)
LG = k2.determinize(LG)
LG = k2.remove_epsilon(LG)
LG = k2.connect(LG)
HLG = k2.compose(H, LG)
```
"""

# ╔═╡ 62bdb386-4d82-474d-9737-b866f031b017
md"""
```math
\begin{align}
	\mathbf{T} = 
\end{align}
```
"""

# ╔═╡ fe1f8289-0457-474c-aa15-3866f7ff8ece
md"""
```math
\begin{align}
	\mathbf{T}^\top \mathbf{s} = \big[ \color{blue} \mathbf{S} 
		\color{black} + \color{green} \mathbf{V} 
		( \color{black} I + \color{orchid} E \color{black})^\top
		\color{red} U^\top \color{black} \big] \mathbf{s}
\end{align}
```
"""

# ╔═╡ eda818d7-be07-4634-8ee3-82f72cfc88c9
md"""
```math
\begin{align}
	\mathbf{T}^\top \mathbf{s} = \big[ \color{blue} \mathbf{S} 
		\color{black} + \color{green} \mathbf{V} 
		( \color{black} I + \color{orchid} E \color{black})^\top
		\color{black} \big] \Big\{ \color{red} U^\top  \color{black} \mathbf{s} \Big\}
\end{align}
```
"""

# ╔═╡ ad404cf2-60b2-4d3a-b0cd-8f86f3399e4d
md"""
```math
\begin{align}
	\mathbf{T}^\top \mathbf{s} = \big[ \color{blue} \mathbf{S} 
		\color{black} + \color{green} \mathbf{V} \color{black} \big]
		\Big\{  ( \color{black} I + \color{orchid} E \color{black})^\top 
		\color{red} U^\top  \color{black} \mathbf{s} \Big\}
\end{align}
```
"""

# ╔═╡ 67138393-e419-4443-9deb-75cac2ad49a2
md"""
```math
\begin{align}
	\mathbf{T}^\top \mathbf{s} = \color{blue} \mathbf{S} 
		\color{black} + \Big\{  \color{green} \mathbf{V}
		 ( \color{black} I + \color{orchid} E \color{black})^\top
		\color{red} U^\top  \color{black} \mathbf{s} \Big\}
\end{align}
```
"""

# ╔═╡ 146e106e-52b1-46bc-af45-4b584970b195
md"""
```math
\begin{align}
	\mathbf{T}^\top \mathbf{s} = \Big\{ \big[ \color{blue} \mathbf{S} 
		\color{black} + \color{green} \mathbf{V}
		 ( \color{black} I + \color{orchid} E \color{black})^\top
		\color{red} U^\top  \color{black} \big] \mathbf{s} \Big\}
\end{align}
```
"""

# ╔═╡ 45cf9d7a-3897-49d9-b963-06d0d139eff3
md"""
```math
\begin{align}
	\mathbf{T} = \mathbf{S} 
		+ \mathbf{U} 
		\bigg( \sum_{i=0}^{k-1} \mathbf{E}^i \bigg)
		V^\top
\end{align}
```
"""

# ╔═╡ 580e697d-8d0d-464c-b73d-5e00228a7317
K = ProbSemiring{Float64}

# ╔═╡ 1a906a8f-243a-4111-a55f-73f22728a6db
f_fsa = FactorizedFSA(
	[1 => one(K), 2 => one(K), 3 => one(K)],
	[(1, 0) => one(K), (2, -1) => one(K), (3, 0) => one(K),
	 (0, 4) => one(K), (0,5) => one(K), (0, -1) => one(K),
	 (-1, 6) => one(K)],
	[4 => one(K), 5 => one(K), 6 => one(K)],
	["bicycle", "car", "water", "bear", "panda", "firefox"]
)

# ╔═╡ e666ca38-1ec8-4bfe-bd46-dbcb84fe8b7d
show(stdout, MIME("dot"), f_fsa)

# ╔═╡ d4a2ac78-f77e-47ea-b0b0-ce2399d06739
MIME"dot"

# ╔═╡ 529121d3-2f06-40f4-8d55-d7aa481b8a4c
str=""

# ╔═╡ 32f1aac0-4dc6-4385-9fd2-6a688d639e89
write(str, "hello")

# ╔═╡ f4425bf2-a279-488e-aff1-2d4decd7505a
rmepsilon(f_fsa)

# ╔═╡ 7ada4422-98d2-415c-a8fd-f2c4cd4224e4
f_fsa isa FactorizedFSA

# ╔═╡ 8ce8ec21-13e0-4bb6-a272-d8778106cc2c
A = SparseFSA(
	[1 => one(K)],
	[(1, 2) => K(4) + one(K), (2, 3) => K(5), (2, 1) => K(4),
	 (1, 3) => one(K)],
	[2 => one(K), 3 => one(K)],
	["a", "b", "c"]
)

# ╔═╡ 6b07528a-40f1-4021-8606-ced802a30ed9
B = SparseFSA(
	[1 => one(K)],
	[(1, 2) => K(3), (2, 3) => one(K), (3, 4) => one(K), (2, 1) => K(2),
	 (1, 5) => K(4), (5, 3) => K(1/2)],
	[4 => one(K)],
	["a", "b", "a", "c", "e"]
)

# ╔═╡ fff93813-a30e-4c8e-b003-85626941b713
FSA(
	kron(A.α, B.α),
	kron(A.T, B.T),
	kron(A.ω, B.ω),
	kron(A.λ, B.λ)
)

# ╔═╡ bd534392-97b9-4f4a-b981-8695018defac
intersect(B, A)

# ╔═╡ 9cae63b2-ac50-4772-a540-4dbf0facdc02
C = DenseForwardFSA(
	K.(randn(3, 5)),
	repeat(["a", "b", "c"], 5)
)

# ╔═╡ 99790beb-e86a-49a1-b241-44a035ad0f45
intersect(A, C)

# ╔═╡ 8d2540cf-846a-4ed9-ae53-87121100c848
FSA(
	kron(A.α, C.α),
	kron(A.T, C.T),
	kron(A.ω, C.ω),
	kron(A.λ, C.λ)
)

# ╔═╡ d8917212-0bd3-40eb-9399-e4f25f883799
kron(A.T, C.T)

# ╔═╡ Cell order:
# ╠═aba927a2-1032-11ed-037f-452f342c54c6
# ╠═093fda8f-4ce7-4cc2-90ee-1bef16a7c915
# ╠═9124ca89-b612-48ab-ba89-1a671265575e
# ╠═597e70a6-9fe7-4168-be1a-cdf17ce67c40
# ╠═ba70bef7-161e-4060-9716-3ed3f8c4616d
# ╠═4ff95290-2872-42a2-8725-e32e730e2fc9
# ╠═09aaf348-fc20-4985-bb7d-c25d04659a8a
# ╠═62bdb386-4d82-474d-9737-b866f031b017
# ╠═fe1f8289-0457-474c-aa15-3866f7ff8ece
# ╠═eda818d7-be07-4634-8ee3-82f72cfc88c9
# ╠═ad404cf2-60b2-4d3a-b0cd-8f86f3399e4d
# ╠═67138393-e419-4443-9deb-75cac2ad49a2
# ╠═146e106e-52b1-46bc-af45-4b584970b195
# ╠═45cf9d7a-3897-49d9-b963-06d0d139eff3
# ╠═580e697d-8d0d-464c-b73d-5e00228a7317
# ╠═1a906a8f-243a-4111-a55f-73f22728a6db
# ╠═e666ca38-1ec8-4bfe-bd46-dbcb84fe8b7d
# ╠═d4a2ac78-f77e-47ea-b0b0-ce2399d06739
# ╠═529121d3-2f06-40f4-8d55-d7aa481b8a4c
# ╠═32f1aac0-4dc6-4385-9fd2-6a688d639e89
# ╠═f4425bf2-a279-488e-aff1-2d4decd7505a
# ╠═7ada4422-98d2-415c-a8fd-f2c4cd4224e4
# ╠═8ce8ec21-13e0-4bb6-a272-d8778106cc2c
# ╠═6b07528a-40f1-4021-8606-ced802a30ed9
# ╠═fff93813-a30e-4c8e-b003-85626941b713
# ╠═bd534392-97b9-4f4a-b981-8695018defac
# ╠═9cae63b2-ac50-4772-a540-4dbf0facdc02
# ╠═99790beb-e86a-49a1-b241-44a035ad0f45
# ╠═8d2540cf-846a-4ed9-ae53-87121100c848
# ╠═d8917212-0bd3-40eb-9399-e4f25f883799
