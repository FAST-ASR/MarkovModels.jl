module MarkovModels

using LinearAlgebra
using SparseArrays
using StatsBase: sample, Weights
using StatsFuns: logaddexp, logsumexp
import Base: union

export ProbabilitySemiField
export MaxTropicalSemiField
export MinTropicalSemiField
export LogSemiField

include("algstruct.jl")

#######################################################################
# FSM definition

export FSM
export LinearFSM
export addstate!
export link!
export initstates
export finalstates
export isinit
export isfinal
export links
export states
export setstart!
export setfinal!

include("fsm.jl")

#######################################################################
# FSM algorithms

export compose
export concat
export determinize
export minimize
export removenilstates
export relabel
export weightnormalize

include("fsmop.jl")

export compile
include("compiledfsm.jl")

#######################################################################
# Algorithms for inference with Markov chains

export αrecursion
export αβrecursion
export βrecursion
export resps
export beststring
export samplestring

include("inference.jl")

#######################################################################
# Pretty display functions

# Pretty display of the FSM in IJulia as a graph.
function Base.show(io, ::MIME"image/svg+xml", fsm::FSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")


    for s in states(fsm)
        name = "$(s.id)"
        label = "label=\"$(s.id)"
        attrs = "penwidth=" * (isinit(s) ? "2" : "1")
        label *= isinit(s) ? "/$(round(s.startweight.val, digits = 3))" : ""
        attrs *= " shape=" * (isfinal(s) ? "doublecircle" : "circle")
        label *= isfinal(s) ? "/$(round(s.finalweight.val, digits = 3))\"" : "\""
        write(dotfile, "$name [ $label $attrs ];\n")
    end

    for src in states(fsm)
        for link in links(fsm, src)
            weight = round(link.weight.val, digits = 3)

            srcname = "$(src.id)"
            destname = "$(link.dest.id)"

            ilabel = isnothing(link.ilabel) ? "ϵ" : link.ilabel
            olabel = isnothing(link.olabel) ? "ϵ" : link.olabel
            lname = "$(ilabel):$(olabel)/"
            lname *= "/$(weight)"
            write(dotfile, "$srcname -> $destname [ label=\"$(lname)\" ];\n")
        end
    end

    write(dotfile, "}\n")
    close(dotfile)
    run(`dot -Tsvg $(dotpath) -o $(svgpath)`)

    xml = read(svgfile, String)
    write(io, xml)

    close(svgfile)

    rm(dotpath)
    rm(svgpath)
end

# Pretty display the sparse matrix (i.e. from αβrecursion).
import Printf:@sprintf
function Base.show(
    io::IO,
    ::MIME"text/plain",
    a::Array{Dict{State, T},1}
) where T <: AbstractFloat

    for n in 1:length(a)
        write(io, "[n = $n]  \t")
        max = first(sort(collect(a[n]), by = x -> x[2], rev = true))
        write(io, "$(first(max).id)")
        for (s, w) in sort(a[n]; by = x -> x.id)
            write(io, "\t$(s.id) = $(@sprintf("%.3f", w))  ")
        end
        write(io, "\n")
    end
end

end

