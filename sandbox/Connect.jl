### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b79aee54-195d-11ed-389d-ff3b4ae31b25
begin
	using Pkg
	Pkg.activate("../")

	using Revise
	using Semirings
	using MarkovModels	
end

# ╔═╡ 0b25b4b1-2197-49e0-9e52-387949d5a106
K = ProbSemiring{Float32}

# ╔═╡ d1781c3c-af8d-4e2c-b830-d9a08abef91d
fsa1 = FSA(
	K[1, 0, 0],
	K[0  3  0; 3 0 3; 0 0 0],
	K[0, 0, 1],
	["a", "b", "c"];
	accessible,
	coaccessible,
	deterministic, 
	lexsorted, 
)

# ╔═╡ 4d0a0657-c316-4566-b99a-cc75498138da
summary(fsa1)

# ╔═╡ d75a30e1-35ee-4798-9166-a084cd501758
§÷ 3

# ╔═╡ 4b2bcc84-e6d6-484a-9e9f-958fbef2ffba
fsa1.α[findall(!iszero, fsa1.α)]

# ╔═╡ efb9b8d7-1a9c-45b2-9490-e65e3a177ea4
fsa2 = FSA(
	K[1, 0, 0, 0],
	K[0  2  0 0; 0 0 2 2; 0 0 0 2; 0 0 0 0],
	K[0, 0, 0, 1],
	["a", "b", "b", "c"];
	accessible,
	acyclic,
	coaccessible,
	deterministic, 
	lexsorted, 
	topsorted,
)

# ╔═╡ 54d81621-fa2d-41fe-a31d-81036a272910
@bind steps html"<input type=\"range\" min=1 max=10>"

# ╔═╡ 44fdc96a-3bb8-487a-983e-77164a70eb83
md"""Iteration step: $steps"""

# ╔═╡ ee72ee2d-c272-44c1-85e3-71a091e7bac1
function browse_nsteps(fsa, steps, iteratortype = FSAIterator)
	itfsa = iteratortype(fsa)
	next = iterate(itfsa)
	step = 0
	sv = nothing
	while next != nothing && step < steps
		(sv, state) = next
		next = iterate(itfsa, state)
		step += 1
	end
	
	sv
end

# ╔═╡ 718091ba-a1b0-4ab3-9e91-71bd118e6124
(
	fsa2, 
	Dict(
		s => "lightblue" 
		for s in MarkovModels.activestates(browse_nsteps(fsa2, steps))[1]
	)
)

# ╔═╡ 62d61dc4-c236-4c97-8930-21bd2ac4ccda
for s in Base.Iterators.filter( -> i==1, 
							   zip(MarkovModels.activestates(fsa2.α)...))
	println("hello")
end

# ╔═╡ 88f273e1-c55a-41ba-b11f-66eee7776ce9
summary(fsa2)

# ╔═╡ c6b50866-e07a-4679-9581-c707bb2385a9
fsa3 = FSA(
	kron(fsa1.α, fsa2.α),
	kron(fsa1.T, fsa2.T),
	kron(fsa1.ω, fsa2.ω),
	kron(fsa1.λ, fsa2.λ)
)

# ╔═╡ Cell order:
# ╠═b79aee54-195d-11ed-389d-ff3b4ae31b25
# ╠═0b25b4b1-2197-49e0-9e52-387949d5a106
# ╠═d1781c3c-af8d-4e2c-b830-d9a08abef91d
# ╠═4d0a0657-c316-4566-b99a-cc75498138da
# ╠═d75a30e1-35ee-4798-9166-a084cd501758
# ╠═4b2bcc84-e6d6-484a-9e9f-958fbef2ffba
# ╠═efb9b8d7-1a9c-45b2-9490-e65e3a177ea4
# ╟─54d81621-fa2d-41fe-a31d-81036a272910
# ╟─44fdc96a-3bb8-487a-983e-77164a70eb83
# ╠═718091ba-a1b0-4ab3-9e91-71bd118e6124
# ╠═ee72ee2d-c272-44c1-85e3-71a091e7bac1
# ╠═62d61dc4-c236-4c97-8930-21bd2ac4ccda
# ╠═88f273e1-c55a-41ba-b11f-66eee7776ce9
# ╠═c6b50866-e07a-4679-9581-c707bb2385a9
