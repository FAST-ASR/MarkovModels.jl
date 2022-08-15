### A Pluto.jl notebook ###
# v0.19.9

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
	using PlutoUI
	using Pkg
	Pkg.activate("../")

	using Revise
	using Semirings
	using MarkovModels
	using GraphViz
	
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
md""" 
Iteration step: $(@bind steps Slider(1:5)) 
"""

# ╔═╡ 718091ba-a1b0-4ab3-9e91-71bd118e6124
steps

# ╔═╡ ee72ee2d-c272-44c1-85e3-71a091e7bac1
begin
	local fsa = fsa1
	local next = iterate(fsa)
	local step = 0
	local sv = nothing
	while next != nothing && step < steps
		(sv, state) = next
		next = iterate(fsa, state)
		step += 1
	end
	
	states, weights = MarkovModels.activestates(sv) 
	fsa, Dict(s => "lightblue" for s in states)
end

# ╔═╡ ca702ebf-6dc1-4ad0-943d-9bcf9343fd59
@bind forward Button("prev")

# ╔═╡ bd74d099-5809-4286-9250-9bafd36e4a84
let
	forward

	md"I am $(rand(1:10)) $forward"
end

# ╔═╡ 03917546-c49f-486c-866f-88a3cb907edb
@bind s Slider(1:10)

# ╔═╡ 14892a5a-fab7-40c4-a3fb-9bfaf566a84d
s

# ╔═╡ b742942b-b2a5-4155-974c-655319b15267
function forward_backward(directions)
	return PlutoUI.combine() do Child
		
		inputs = [
			md""" $( Child(name, Button("prev")) )"""
			
			for name in directions
		]
		
		md"""
		#### Wind speeds
		$( Child("prev", Button("prev")) ) 
		"""
	end
end

# ╔═╡ 3561e52d-4113-4432-8b60-b98158449440
@bind buttons forward_backward()

# ╔═╡ 54583ff1-2f4b-44ed-b4d0-63821437a79a
@bind fb forward_backward(["North", "South"])

# ╔═╡ 0420a435-d739-4b06-addf-8b7acacce32b
tokens = collect(AcyclicIterator(fsa1))

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
# ╠═4b2bcc84-e6d6-484a-9e9f-958fbef2ffba
# ╠═efb9b8d7-1a9c-45b2-9490-e65e3a177ea4
# ╠═54d81621-fa2d-41fe-a31d-81036a272910
# ╠═718091ba-a1b0-4ab3-9e91-71bd118e6124
# ╠═ee72ee2d-c272-44c1-85e3-71a091e7bac1
# ╠═3561e52d-4113-4432-8b60-b98158449440
# ╠═ca702ebf-6dc1-4ad0-943d-9bcf9343fd59
# ╠═bd74d099-5809-4286-9250-9bafd36e4a84
# ╠═03917546-c49f-486c-866f-88a3cb907edb
# ╠═14892a5a-fab7-40c4-a3fb-9bfaf566a84d
# ╠═54583ff1-2f4b-44ed-b4d0-63821437a79a
# ╠═b742942b-b2a5-4155-974c-655319b15267
# ╠═0420a435-d739-4b06-addf-8b7acacce32b
# ╠═62d61dc4-c236-4c97-8930-21bd2ac4ccda
# ╠═88f273e1-c55a-41ba-b11f-66eee7776ce9
# ╠═c6b50866-e07a-4679-9581-c707bb2385a9
