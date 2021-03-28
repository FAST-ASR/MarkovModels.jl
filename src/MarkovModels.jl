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
export initstate
export finalstate
export isinit
export isfinal
export links
export states

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

    write(dotfile, "s [ shape=circle label=\"<s>\" penwidth=2 fixedsize=true width=0.6 ];\n")
    write(dotfile, "e [ shape=doublecircle label=\"</s>\" penwidth=1 fixedsize=true width=0.6 ];\n")


    for s in states(fsm)
        (isinit(s) || isfinal(s)) && continue
        attrs = ""
        name = ""
        if  isemitting(s)
            name = "$(s.id)"
            attrs *=  "shape=circle"
            attrs *= " label=\"$(s.pdfindex)\""
            attrs *= " style=filled fillcolor=" * (isemitting(s) ? "lightblue" : "none")
        else
            name = "$(s.id)"
            attrs *= "shape=circle label=\"$(s.id)\""
        end
        write(dotfile, "$name [ $attrs ];\n")
    end

    for src in states(fsm)
        for link in links(fsm, src)
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

            lname = ""
            if hasinputlabel(link) && hasoutputlabel(link)
                #lname *= link.ilabel == link.olabel ?  "$(link.ilabel)/" : "$(link.ilabel):$(link.olabel)/"
                lname *= "$(link.ilabel):$(link.olabel)/"
            elseif hasinputlabel(link)
                lname *= "$(link.ilabel):ϵ/"
            elseif hasoutputlabel(link)
                lname *= "ϵ:$(link.olabel)/"
            end
            lname *= "$(weight)"
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
