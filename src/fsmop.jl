# Implementation of common FSM operations.

"""
    transpose(fsm)

Transpose the fsm, i.e. reverse all it's arcs. The final state becomes
the initial state.
"""
function Base.transpose(fsm::FSM{T}) where T<:SemiField
    nfsm = FSM{T}()
    smap = Dict(initstateid => finalstate(nfsm), finalstateid => initstate(nfsm))
    for s in states(fsm)
        (isinit(s) || isfinal(s)) && continue
        smap[s.id] = addstate!(nfsm, id = s.id, pdfindex = s.pdfindex)
    end

    for src in states(fsm)
        for l in links(fsm, src)
            link!(nfsm, smap[l.dest.id], smap[src.id], ilabel = l.ilabel,
                  olabel = l.olabel, weight = l.weight)
        end
    end
    nfsm
end

"""
    union(fsm1, fsm2, ...)
    ∪(fsm1, fsm2, ...)

Append severall FSMs to form a single one.
"""
function Base.union(fsm1::FSM{T}, fsm2::FSM{T}) where T<:SemiField
    fsm = FSM{T}()

    smap = Dict{State, State}(initstate(fsm1) => initstate(fsm),
                              finalstate(fsm1) => finalstate(fsm))
    for s in states(fsm1)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex)
    end

    for src in states(fsm1)
        for l in links(fsm1, src)
            link!(fsm, smap[src], smap[l.dest], ilabel = l.ilabel,
                  olabel = l.olabel, weight = l.weight)
        end
    end

    smap = Dict{State, State}(initstate(fsm2) => initstate(fsm),
                              finalstate(fsm2) => finalstate(fsm))
    for s in states(fsm2)
        if s.id == finalstateid || s.id == initstateid continue end
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex)
    end

    for src in states(fsm2)
        for l in links(fsm2, src)
                link!(fsm, smap[src], smap[l.dest], ilabel = l.ilabel,
                      olabel = l.olabel, weight = l.weight)
        end
    end

    fsm
end
Base.union(fsm::FSM{T}, rest::FSM{T}...) where T<:SemiField = foldl(union, rest, init=fsm)

"""
    concat(fsm1, fsm2, ...)

Concatenate several FSMs together.
"""
function concat(fsm1::FSM{T}, fsm2::FSM{T}) where T<:SemiField
    fsm = FSM{T}()

    cs = addstate!(fsm) # special non-emitting state for concatenaton
    smap = Dict(initstate(fsm1) => initstate(fsm), finalstate(fsm1) => cs)
    for s in states(fsm1)
        (isinit(s) || isfinal(s)) && continue
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex)
    end

    for src in states(fsm1)
        for l in links(fsm1, src)
            link!(smap[src], smap[l.dest], ilabel = l.ilabel,
                  olabel = l.olabel, l.weight)
        end
    end

    smap = Dict(initstate(fsm2) => cs, finalstate(fsm2) => finalstate(fsm))
    for s in states(fsm2)
        (isinit(s) || isfinal(s)) && continue
        smap[s] = addstate!(fsm, pdfindex = s.pdfindex)
    end

    for src in states(fsm2)
        for l in links(fsm2, src)
            link!(smap[src], smap[l.dest], ilabel = l.ilabel,
                  olabel = l.olabel, weight = l.weight)
        end
    end

    fsm
end
concat(fsm1::FSM{T}, rest::FSM{T}...) where T<:SemiField = foldl(concat, rest, init=fsm1)

"""
    weightnormalize(fsm)

Change the weight of the links such that the sum of the exponentiated
weights of the outgoing links from one state will sum up to one.
"""
function weightnormalize(fsm::FSM{T}) where T<:SemiField
    nfsm = FSM{T}()
    smap = Dict(
        initstateid => initstate(nfsm),
        finalstateid => finalstate(nfsm)
    )
    for s in states(fsm)
        (isinit(s) || isfinal(s)) && continue
        smap[s.id] = addstate!(nfsm, pdfindex = s.pdfindex)
    end

    for src in states(fsm)
        total = zero(T)
        for l in links(fsm, src) total += l.weight end
        for l in links(fsm, src)
            link!(nfsm, smap[src.id], smap[l.dest.id], ilabel = l.ilabel,
                  olabel = l.olabel, weight = l.weight / total)
        end
    end
    nfsm
end

#######################################################################
# FSM determinization

"""
    determinize(fsm)

Transform `fsm` such that each state has at most one link to any other
states.
"""
function determinize(fsm::FSM{T}) where T
    nfsm = FSM{T}()
    smap = Dict()
    newlinks = Set()

    first = Set([(initstate(fsm), one(T))])
    visited = Set()
    queue = Set([first])
    smap[first] = initstate(nfsm)
    while ! isempty(queue)
        state = pop!(queue)
        labels = Dict()
        for (p,v) in state
            for l in links(fsm, p)
                k = (l.ilabel, l.olabel)
                nexts, w, hasfinal = get(labels, k, (Set(), zero(T), false))
                push!(nexts, (l.dest, v*l.weight))
                labels[k] = (nexts, w + v*l.weight, hasfinal || isfinal(l.dest))
            end
        end

        for (label, (nexts, w, hasfinal)) in labels
            newstate = Set()
            for (next, nw) in nexts
                push!(newstate, (next, nw / w))
            end
            push!(newlinks, (state, newstate, label, w))
            if newstate ∉ visited
                push!(visited, newstate)
                if hasfinal
                    smap[newstate] = finalstate(nfsm)
                else
                    smap[newstate] = addstate!(nfsm)
                end
                push!(queue, newstate)
            end
        end
    end

    for (s, ns, label, w) in newlinks
        link!(nfsm, smap[s], smap[ns], ilabel = label[1],
              olabel = label[2], weight = w)
    end
    nfsm
end

"""
    minimize(fsm)

Merge equivalent states to reduce the size of the FSM. Only
the states that have the same `pdfindex` and the same `label` can be
potentially merged.

!!! warning
    The input FSM should not contain cycles otherwise the algorithm
    will never end.
"""
minimize(fsm::FSM) = (transpose ∘ determinize ∘ transpose ∘ determinize)(fsm)

#######################################################################
# NIL state removal

"""
    removenilstates(fsm)

Remove all states that are non-emitting and have no labels (except the
the initial and final states)
"""
function removenilstates(fsm::FSM{T}) where T<:SemiField
    nfsm = FSM{T}()

    newstates = Dict(initstate(fsm) => initstate(nfsm),
                     finalstate(fsm) => finalstate(nfsm))
    for state in states(fsm)
        (! islabeled(state) && ! isemitting(state)) && continue
        newstates[state] = addstate!(nfsm, pdfindex = state.pdfindex,
                                     label = state.label)
    end

    newlinks = Dict()
    stack = [(initstate(fsm), initstate(fsm), one(T))]
    visited = Set([initstate(fsm)])
    while ! isempty(stack)
        src, state, weight = popfirst!(stack)
        for link in links(state)
            if link.dest ∈ keys(newstates)
                link!(newstates[src], newstates[link.dest], weight * link.weight)
                if link.dest ∉ visited
                    push!(visited, link.dest)
                    push!(stack, (link.dest, link.dest, one(T)))
                end
            else
                push!(stack, (src, link.dest, weight * link.weight))
            end
        end
    end

    nfsm
end

"""
    relabel(fsm, ilabelmap, olabelmap)

Replace the labels given maps.
"""
function relabel(fsm::FSM{T}, ilabelmap, olabelmap) where T<:SemiField
    nfsm = FSM{T}()
    smap = Dict()

    for state in states(fsm)
        if isinit(state)
            smap[state] = initstate(nfsm)
        elseif isfinal(state)
            smap[state] = finalstate(nfsm)
        else
            smap[state] = addstate!(nfsm)
        end
    end

    for src in states(fsm)
        for l in links(fsm, src)
            link!(nfsm, smap[src], smap[l.dest],
                  ilabel = get(ilabelmap, l.ilabel, l.ilabel),
                  olabel = get(olabelmap, l.olabel, l.olabel),
                  weight = l.weight)
        end
    end
    nfsm
end

#######################################################################
# Composition

"""
    compose(fsm1, fsm2)
    Base.:∘(fsm1, fsm2)

Replace each state `s` in `fsm` by a "subfsms" from `subfsms` with
associated label `s.label`. `subfsms` should be a Dict{<:Label, FSM}`.
"""
function compose(fsm1::FSM{T}, fsm2::FSM{T}) where T
    fsm = FSM{T}()
    smap = Dict()
    newlinks = Set()

    visited = Set()
    queue = [(initstate(fsm1), initstate(fsm2))]

    while ! isempty(queue)
        newstate = popfirst!(queue)

        if isinit(newstate[1]) && isinit(newstate[2])
            smap[newstate] =  initstate(fsm)
        elseif isfinal(newstate[1]) && isfinal(newstate[2])
            smap[newstate] = finalstate(fsm)
        else
            smap[newstate] = addstate!(fsm)
        end

        for e₁ in links(fsm1, newstate[1])
            for e₂ in links(fsm2, newstate[2])
                e₁.olabel == e₂.ilabel || continue
                #(e₁.olabel == e₂.ilabel || isnothing(e₁.olabel) || isnothing(e₂.ilabel)) || continue
                nextstate = (e₁.dest, e₂.dest)
                if nextstate ∉ visited
                    push!(visited, nextstate)
                    push!(queue, nextstate)
                end
                push!(newlinks, (newstate, nextstate,
                     ilabel = e₁.ilabel, olabel = e₂.olabel,
                     weight = e₁.weight * e₂.weight))
            end
        end
    end

    for (src, dest, ilabel, olabel, weight) in newlinks
        link!(fsm, smap[src], smap[dest]; ilabel, olabel, weight)
    end

    fsm
end
Base.:∘(fsm1::FSM, fsm2::FSM) = compose(fsm1, fsm2)

#function Base.convert(F::Type{FSM{T}}, fsm::FSM{T2}) where {T<:SemiField,T2<:SemiField}
#    nfsm = F()
#
#    smap = Dict(initstate(fsm) => initstate(nfsm), finalstate(fsm) => finalstate(nfsm))
#    for s in states(fsm)
#        (isinit(s) || isfinal(s)) && continue
#        newstate = addstate!(nfsm, id = s.id, pdfindex = s.pdfindex)
#        smap[s] = newstate
#    end
#
#    for src in states(fsm)
#        for l in links(src)
#            link!(smap[src], smap[l.dest], convert(T, l.weight))
#        end
#    end
#
#    nfsm
#end

