### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 97f84630-1d8d-11ed-2e97-3dd50b3514bb
begin
	using Pkg
	Pkg.activate("../")

	using Random
	using Revise
	using MarkovModels
end

# ╔═╡ 9810558d-49ca-4434-9be8-2ff70de85d37
I₀, J₀, V₀ = [1, 2, 2, 3, 4], [1, 2, 3, 2, 3], [5, 8, 5, 6, 3]

# ╔═╡ 2ebc1134-0bb4-4709-a7eb-cccdc963a638
m, n = 4, 4

# ╔═╡ f8267fe9-3a25-43a6-b63a-b795d4a228f1
rp = randperm(length(V₀))

# ╔═╡ abbd53d1-bc67-45bc-9938-37c5b42344d8
I, J, V = I₀[rp], J₀[rp], V₀[rp]

# ╔═╡ baac2ee8-708c-4d7f-ad9d-f5587e0799d8
p = sortperm(I)

# ╔═╡ 54a9d189-3fc1-444f-a677-29cf13418d83
rowptr = similar(I₀, m + 1)

# ╔═╡ e798ccb8-4685-4d46-a958-f820f747ca1a
begin
	nzsofar = 0
	current_row = 0
	for i in I₀
		if i != current_row
			rowptr[i] = nzsofar + 1
			current_row = i
		end
		nzsofar += 1
	end
	rowptr[m+1] = nzsofar + 1
	
	rowptr
end

# ╔═╡ db673ce1-b7a1-4ea1-b335-408ee6b8a5b9
V₀[rowptr[3]:(rowptr[3+1]-1)]

# ╔═╡ Cell order:
# ╠═97f84630-1d8d-11ed-2e97-3dd50b3514bb
# ╠═9810558d-49ca-4434-9be8-2ff70de85d37
# ╠═2ebc1134-0bb4-4709-a7eb-cccdc963a638
# ╠═f8267fe9-3a25-43a6-b63a-b795d4a228f1
# ╠═abbd53d1-bc67-45bc-9938-37c5b42344d8
# ╠═baac2ee8-708c-4d7f-ad9d-f5587e0799d8
# ╠═54a9d189-3fc1-444f-a677-29cf13418d83
# ╠═e798ccb8-4685-4d46-a958-f820f747ca1a
# ╠═db673ce1-b7a1-4ea1-b335-408ee6b8a5b9
