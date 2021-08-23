# SPDX-License-Identifier: MIT

function generate_strings(fsm::AbstractFSM)
    final_strings = []
    queue = [([s1.label], s1, s1.initweight, Set([s1]))
             for s1 in filter(isinit, states(fsm))]
    while ! isempty(queue)
        strings, s, w, visited = popfirst!(queue)

        if isfinal(s)
            for str in strings
                push!(final_strings, (str, w*s.finalweight))
            end
        end

        for a in arcs(fsm, s)
            if a.dest ∉ visited
                push!(visited, a.dest)
                newstrings = [str*a.dest.label for str in strings]
                push!(queue, (newstrings, a.dest, w*a.weight, visited))
            end
        end
    end

    final_strings
end

function fsmequal(fsm1::AbstractFSM, fsm2::AbstractFSM)
    ws1 = generate_strings(fsm1)
    ws2 = generate_strings(fsm2)

    if length(ws1) ≠ length(ws2)
        return false
    end

    same_strings = Set([t[1] for t in ws1]) == Set([t[1] for t in ws2])
    same_weights = all(sort([convert(Float64, t[2]) for t in ws1]) .≈
                       sort([convert(Float64, t[2]) for t in ws2]))
    same_strings && same_weights
end

@testset "renormalize" begin
    SR = ProbabilitySemifield{Float32}
    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a", initweight = SR(0.5))
    s2 = addstate!(fsm, "b", initweight = SR(0.5))
    s3 = addstate!(fsm, "c", finalweight = SR(0.25))
    addarc!(fsm, s1, s1, SR(0.75))
    addarc!(fsm, s1, s2, SR(0.25))
    addarc!(fsm, s2, s2, SR(0.75))
    addarc!(fsm, s2, s3, SR(0.25))
    addarc!(fsm, s3, s3, SR(0.75))

    @test fsmequal(fsm, fsm |> renormalize)

    rfsm = VectorFSM{SR}()
    s1 = addstate!(rfsm, "a", initweight = SR(1))
    s2 = addstate!(rfsm, "b", initweight = SR(1))
    s3 = addstate!(rfsm, "c", finalweight = SR(0.5))
    addarc!(rfsm, s1, s1, SR(1.5))
    addarc!(rfsm, s1, s2, SR(0.5))
    addarc!(rfsm, s2, s2, SR(1.5))
    addarc!(rfsm, s2, s3, SR(0.5))
    addarc!(rfsm, s3, s3, SR(1.5))

    @test fsmequal(fsm, rfsm |> renormalize)
end

#=
@testset "determinize" begin
    fsm = FSM()

    s1 = addstate!(fsm, label = "a", pdfindex = 1)
    s2 = addstate!(fsm)
    s3 = addstate!(fsm, label = "c", pdfindex = 2)
    setinit!(s1)
    setfinal!(s3)

    addarc!(fsm, s1, s2)
    addarc!(fsm, s2, s3)
    addarc!(fsm, s3, s1)

    @test_throws MarkovModels.InvalidFSMError determinize(fsm)
end
=#
