module MarkovModels

using LinearAlgebra
using SparseArrays
using StatsBase: sample, Weights
using StatsFuns: logaddexp, logsumexp
import Base: union

export ProbabilitySemiField
export MaxTropicalSemiField
export LogSemiField

include("algstruct.jl")

#######################################################################
# FSM definition

# Forward definition of the Abs. state/link to avoid issue with
# circular dependencies.
abstract type AbstractState end
abstract type AbstractLink{T} end

export AbstractLink
export Link

include("link.jl")

export PdfIndex
export Label
export State
export InitStateID
export initstateid
export FinalStateID
export finalstateid

export isemitting
export isinit
export isfinal
export islabeled
export nextemittingstates

include("state.jl")

export FSM
export LinearFSM
export addstate!
export link!
export removestate!
export unlink!

export initstate
export finalstate
export emittingstates
export links
export states

include("fsm.jl")

#######################################################################
# FSM algorithms

export compile
export compose
export concat
export determinize
export minimize
export removenilstates
export weightnormalize

include("fsmop.jl")

include("compiledfsm.jl")

#######################################################################
# Pruning strategies

export PruningStrategy
export BackwardPruning
export CompoundPruning
export SafePruning
export ThresholdPruning
export nopruning

#include("pruning.jl")

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
        attrs = ""
        name = ""
        if islabeled(s) || isemitting(s)
            name = "$(s.id)"
            attrs *=  "shape=circle"
            attrs *= " label=\"" * (islabeled(s) ? "$(s.label)" : "$(s.pdfindex)") * "\""
            attrs *= " style=filled fillcolor=" * (isemitting(s) ? "lightblue" : "none")
        elseif isfinal(s) || isinit(s)
            name = isinit(s) ? "s" : "e"
            attrs *= " shape=" * (isfinal(s) ? "doublecircle" : "circle")
            attrs *= " label=" * (isfinal(s) ? "\"</s>\"" : "\"<s>\"")
            attrs *= " penwidth=" * (isinit(s) ? "2" : "1")
            attrs *= " fixedsize=true width=0.6"
        else
            name = "$(s.id)"
            attrs *= "shape=point"
        end
        write(dotfile, "$name [ $attrs ];\n")
    end

    for src in states(fsm)
        for link in links(src)
            weight = round(link.weight.val, digits = 3)

            srcname = ""
            if isinit(src)
                srcname = "s"
            elseif isfinal(src)
                srcname = "e"
            else
                srcname = "$(src.id)"
            end

            destname = ""
            if isinit(link.dest)
                destname = "s"
            elseif isfinal(link.dest)
                destname = "e"
            else
                destname = "$(link.dest.id)"
            end

            write(dotfile, "$srcname -> $destname [ label=\"$(weight)\" ];\n")
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
