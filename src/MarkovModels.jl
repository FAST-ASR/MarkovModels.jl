
module MarkovModels

using StatsFuns: logaddexp, logsumexp
import Base: union

#######################################################################
# FSM definition

export StateID
export initstateid
export finalstateid

export Label
export Pdfindex

export Link

export State
export isemitting
export isinit
export isfinal

export FSM
export LinearFSM
export addstate!
export link!
export removestate!
export unlink!

export initstate
export finalstate
export name

export backward
export children
export emittingstates
export forward
export links
export parents
export states

include("fsm.jl")

#######################################################################
# FSM algorithms

export addselfloop!
export compose!
export concat
export determinize!
export minimize!
export removenilstates!
export weightnormalize!

include("fsmop.jl")

#######################################################################
# Algorthms for inference with Markov chains

export PruningStrategy
export ThresholdPruning
export nopruning

# Baum-Welch algorithm
export αrecursion
export αβrecursion
export βrecursion
export resps
export viterbi

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
            attrs *= " label=\"" * (islabeled(s) ? s.label : s.pdfindex) * "\""
            attrs *= " style=filled fillcolor=" * (isemitting(s) ? "lightblue" : "none")
        elseif isfinal(s) || isinit(s)
            name = isinit(s) ? "s" : "e"
            attrs *= " shape=" * (isfinal(s) ? "doublecircle" : "circle")
            attrs *= " label=" * (isfinal(s) ? "\"</s>\"" : "\"<s>\"")
            attrs *= " penwidth=" * (isinit(s) ? "2" : "1")
            attrs *= " fixedsize=true width=0.6"
        else
            attrs *= "shape=point"
        end

        write(dotfile, "$name [ $attrs ];\n")
    end

    for link in links(fsm)
        weight = round(link.weight, digits = 3)

        srcname = ""
        if isinit(link.src)
            srcname = "s"
        elseif isfinal(link.src)
            srcname = "e"
        else
            srcname = "$(link.src.id)"
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
        max = foldl(((sa,wa), (s,w)) -> wa < w ? (s,w) : (sa,wa), a[n]; init=first(a[n]))
        write(io, "$(first(max).id)")
        for (s, w) in sort(a[n]; by = x -> x.id)
            write(io, "\t$(s.id) = $(@sprintf("%.3f", w))  ")
        end
        write(io, "\n")
    end
end

#######################################################################
# Other

#include("../src/misc.jl")

end
