# SPDX-License-Identifier: MIT

"""
    Label(x)

Create a FSM label.
"""
Label(x) = UnionConcatSemiring(Set([SymbolSequence([x])]))

function Base.union(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α, fsm2.α),
        BlockDiagonal([fsm1.T, fsm2.T]),
        vcat(fsm1.ω, fsm2.ω),
        vcat(fsm1.λ, fsm2.λ)
    )
end

function concat(fsm1::FSM{K}, fsm2::FSM{K}) where K
    FSM(
        vcat(fsm1.α, zero(fsm2.α)),
        [fsm1.T       fsm1.ω * fsm2.α';
         zero(fsm2.ω * fsm1.α')     fsm2.T],
        vcat(zero(fsm1.ω), fsm2.ω),
        vcat(fsm1.λ, fsm2.λ)
    )
end

function Base.adjoint(fsm::FSM)
    FSM(fsm.ω, fsm.T', fsm.α, fsm.λ)
end

function renorm(fsm::FSM)
    Z = sum(fsm.T, dims=2) .+ fsm.ω
    FSM(
        fsm.α ./ sum(fsm.α),
        fsm.T ./ Z,
        fsm.ω ./ dropdims(Z, dims=2),
        fsm.λ
    )
end

function _mapping_matrix(K::Type{<:Semiring}, fsm₁, fsms)
    blocks = []
    for fsmⁱ in fsms
        n = length(fsmⁱ.α)
        push!(blocks, sparse(1:n, ones(n), ones(K, n)))
    end
    blockdiag(blocks...)
end

function compose(fsm₁::FSM, fsms::AbstractVector{<:FSM{K}},
                 sep = Label(:(:))) where K
    ω = vcat([fsmⁱ.ω for fsmⁱ in fsms]...)
    α = vcat([fsmⁱ.α for fsmⁱ in fsms]...)
    Mₖ = _mapping_matrix(K, fsm₁, fsms)
    T₂ = BlockDiagonal([fsmⁱ.T for fsmⁱ in fsms]) + (Mₖ * fsm₁.T * Mₖ') .* (ω * α')
    FSM(
        vcat([α₁ᵢ * fsmⁱ.α for (α₁ᵢ, fsmⁱ) in zip(fsm₁.α, fsms)]...),
        T₂,
        vcat([ω₁ᵢ * fsmⁱ.ω for (ω₁ᵢ, fsmⁱ) in zip(fsm₁.ω, fsms)]...),
        vcat([λ₁ᵢ * sep * fsmⁱ.λ for (λ₁ᵢ, fsmⁱ) in zip(fsm₁.λ, fsms)]...)
    )
end
Base.:∘(fsm₁::FSM, fsms::AbstractVector) = compose(fsm₁, fsms)
