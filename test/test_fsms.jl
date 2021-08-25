# SPDX-License-Identifier: MIT

function mergelabels(l1, l2)
    if typeof(l1) <: Tuple
        return tuple([mergelabels(x,y) for (x,y) in zip(l1, l2)]...)
    end
    return l1 * l2
end

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
                newstrings = [mergelabels(str, a.dest.label) for str in strings]
                push!(queue, (newstrings, a.dest, w*a.weight, Set(visited)))
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

@testset "union" begin
    SR = LogSemifield{Float32}

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a"; initweight = one(SR), finalweight = one(SR))
    s2 = addstate!(fsm, "b")
    s3 = addstate!(fsm, "b")
    s4 = addstate!(fsm, "c", finalweight = one(SR))
    addarc!(fsm, s1, s2)
    addarc!(fsm, s1, s3)
    addarc!(fsm, s2, s1)
    addarc!(fsm, s1, s4)

    ufsm = VectorFSM{SR}()
    for i in 1:2
        s1 = addstate!(ufsm, "a"; initweight = one(SR), finalweight = one(SR))
        s2 = addstate!(ufsm, "b")
        s3 = addstate!(ufsm, "b")
        s4 = addstate!(ufsm, "c", finalweight = one(SR))

        addarc!(ufsm, s1, s2)
        addarc!(ufsm, s1, s3)
        addarc!(ufsm, s2, s1)
        addarc!(ufsm, s1, s4)
    end

    @test fsmequal(ufsm, union(fsm, fsm))

end

@testset "determinize" begin
    SR = ProbabilitySemifield{Float64}

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
    addarc!(fsm, s1, s1, SR(0.75))
    addarc!(fsm, s1, s2, SR(0.25))
    addarc!(fsm, s2, s2, SR(0.75))
    addarc!(fsm, s2, s3, SR(0.25))
    addarc!(fsm, s3, s3, SR(0.75))
    @test fsmequal(fsm, fsm |> determinize)

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a"; initweight = one(SR), finalweight = one(SR))
    s2 = addstate!(fsm, "b")
    s3 = addstate!(fsm, "b")
    s4 = addstate!(fsm, "c", finalweight = one(SR))
    addarc!(fsm, s1, s2)
    addarc!(fsm, s1, s3)
    addarc!(fsm, s2, s1)
    addarc!(fsm, s1, s4)
    @test fsmequal(fsm |> renormalize, fsm |> determinize)
end

@testset "minimize" begin
    SR = ProbabilitySemifield{Float64}

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
    s4 = addstate!(fsm, "a"; initweight = SR(0.5))
    s5 = addstate!(fsm, "d")
    s6 = addstate!(fsm, "c"; finalweight = SR(0.25))
    s7 = addstate!(fsm, "e")
    addarc!(fsm, s1, s1, SR(0.75))
    addarc!(fsm, s1, s2, SR(0.25))
    addarc!(fsm, s2, s2, SR(0.75))
    addarc!(fsm, s2, s3, SR(0.25))
    addarc!(fsm, s3, s3, SR(0.75))
    addarc!(fsm, s4, s4, SR(0.75))
    addarc!(fsm, s4, s5, SR(0.25))
    addarc!(fsm, s5, s5, SR(0.75))
    addarc!(fsm, s5, s6, SR(0.25))
    addarc!(fsm, s6, s6, SR(0.75))
    addarc!(fsm, s7, s7, SR(0.75))
    addarc!(fsm, s4, s7, SR(0.75))
    addarc!(fsm, s7, s3, SR(0.25))

    mfsm = VectorFSM{SR}()
    s1 = addstate!(mfsm, "a"; initweight = SR(1.0))
    s2 = addstate!(mfsm, "b"; initweight = SR(0.5))
    s3 = addstate!(mfsm, "e")
    s4 = addstate!(mfsm, "d")
    s5 = addstate!(mfsm, "c"; finalweight = SR(0.5))
    addarc!(mfsm, s1, s1, SR(1.5))
    addarc!(mfsm, s1, s2, SR(0.25))
    addarc!(mfsm, s1, s3, SR(0.75))
    addarc!(mfsm, s1, s4, SR(0.25))
    addarc!(mfsm, s2, s2, SR(0.75))
    addarc!(mfsm, s2, s5, SR(0.25))
    addarc!(mfsm, s3, s3, SR(0.75))
    addarc!(mfsm, s3, s5, SR(0.25))
    addarc!(mfsm, s4, s4, SR(0.75))
    addarc!(mfsm, s4, s5, SR(0.25))
    addarc!(mfsm, s5, s5, SR(1.5))
    mfsm = mfsm |> renormalize

    @test fsmequal(mfsm, fsm |> minimize)
end

@testset "transpose" begin
    SR = ProbabilitySemifield{Float64}

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
    addarc!(fsm, s1, s1, SR(0.75))
    addarc!(fsm, s1, s2, SR(0.25))
    addarc!(fsm, s2, s2, SR(0.75))
    addarc!(fsm, s2, s3, SR(0.25))
    addarc!(fsm, s3, s3, SR(0.75))

    rfsm = VectorFSM{SR}()
    s1 = addstate!(rfsm, "a"; finalweight = SR(0.5))
    s2 = addstate!(rfsm, "b"; finalweight = SR(0.5))
    s3 = addstate!(rfsm, "c"; initweight = SR(0.25))
    addarc!(rfsm, s1, s1, SR(0.75))
    addarc!(rfsm, s2, s1, SR(0.25))
    addarc!(rfsm, s2, s2, SR(0.75))
    addarc!(rfsm, s3, s2, SR(0.25))
    addarc!(rfsm, s3, s3, SR(0.75))
    @test fsmequal(rfsm, fsm |> transpose)
    @test fsmequal(fsm, fsm |> transpose |> transpose)
end

@testset "HierarchicalFSM" begin
    SR = ProbabilitySemifield{Float64}

    fsma = VectorFSM{SR}()
    s1 = addstate!(fsma, "a"; initweight = SR(0.5))
    s2 = addstate!(fsma, "b"; initweight = SR(0.5))
    s3 = addstate!(fsma, "c"; finalweight = SR(0.25))
    addarc!(fsma, s1, s1, SR(0.75))
    addarc!(fsma, s1, s2, SR(0.25))
    addarc!(fsma, s2, s2, SR(0.75))
    addarc!(fsma, s2, s3, SR(0.25))
    addarc!(fsma, s3, s3, SR(0.75))

    fsmb = VectorFSM{SR}()
    s1 = addstate!(fsmb, "d"; initweight = SR(0.5))
    s2 = addstate!(fsmb, "e"; initweight = SR(0.5))
    s3 = addstate!(fsmb, "f"; finalweight = SR(0.25))
    addarc!(fsmb, s1, s1, SR(0.75))
    addarc!(fsmb, s1, s2, SR(0.25))
    addarc!(fsmb, s2, s2, SR(0.75))
    addarc!(fsmb, s2, s3, SR(0.25))
    addarc!(fsmb, s3, s3, SR(0.75))

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "x"; initweight = one(SR))
    s2 = addstate!(fsm, "y"; initweight = one(SR))
    s3 = addstate!(fsm, "z"; finalweight = one(SR))
    addarc!(fsm, s1, s2)
    addarc!(fsm, s2, s3)
    addarc!(fsm, s3, s1)
    addarc!(fsm, s1, s3)
    fsm = fsm |> renormalize

    hfsm = VectorFSM{SR}()
    s1 = addstate!(hfsm, ("x", "a"); initweight = SR(0.25))
    s2 = addstate!(hfsm, ("x", "b"); initweight = SR(0.25))
    s3 = addstate!(hfsm, ("x", "c"))
    s4 = addstate!(hfsm, ("y", "d"); initweight = SR(0.25))
    s5 = addstate!(hfsm, ("y", "e"); initweight = SR(0.25))
    s6 = addstate!(hfsm, ("y", "f"))
    s7 = addstate!(hfsm, ("z", "a"))
    s8 = addstate!(hfsm, ("z", "b"))
    s9 = addstate!(hfsm, ("z", "c"); finalweight = SR(0.125))
    addarc!(hfsm, s1, s1, SR(0.75))
    addarc!(hfsm, s1, s2, SR(0.25))
    addarc!(hfsm, s2, s2, SR(0.75))
    addarc!(hfsm, s2, s3, SR(0.25))
    addarc!(hfsm, s3, s3, SR(0.75))
    addarc!(hfsm, s3, s4, SR(0.0625))
    addarc!(hfsm, s3, s5, SR(0.0625))
    addarc!(hfsm, s3, s7, SR(0.0625))
    addarc!(hfsm, s3, s8, SR(0.0625))
    addarc!(hfsm, s4, s4, SR(0.75))
    addarc!(hfsm, s4, s5, SR(0.25))
    addarc!(hfsm, s5, s5, SR(0.75))
    addarc!(hfsm, s5, s6, SR(0.25))
    addarc!(hfsm, s6, s6, SR(0.75))
    addarc!(hfsm, s6, s7, SR(0.125))
    addarc!(hfsm, s6, s8, SR(0.125))
    addarc!(hfsm, s7, s7, SR(0.75))
    addarc!(hfsm, s7, s8, SR(0.25))
    addarc!(hfsm, s8, s8, SR(0.75))
    addarc!(hfsm, s8, s9, SR(0.25))
    addarc!(hfsm, s9, s9, SR(0.75))
    addarc!(hfsm, s9, s1, SR(0.0625))
    addarc!(hfsm, s9, s2, SR(0.0625))

    smap = Dict("x" => fsma, "y" => fsmb, "z" => fsma)
    @test fsmequal(hfsm, HierarchicalFSM(fsm, smap))
end

@testset "MatrixFSM" begin
    SR = ProbabilitySemifield{Float64}

    fsm = VectorFSM{SR}()
    s1 = addstate!(fsm, "a"; initweight = SR(0.5))
    s2 = addstate!(fsm, "b"; initweight = SR(0.5))
    s3 = addstate!(fsm, "c"; finalweight = SR(0.25))
    addarc!(fsm, s1, s1, SR(0.75))
    addarc!(fsm, s1, s2, SR(0.25))
    addarc!(fsm, s2, s2, SR(0.75))
    addarc!(fsm, s2, s3, SR(0.25))
    addarc!(fsm, s3, s3, SR(0.75))

    pdfid_mapping = Dict(
        "a" => 1,
        "b" => 2,
        "c" => 3
    )
    mfsm = MatrixFSM(fsm, pdfid_mapping)

    @test fsmequal(fsm, mfsm)

    ufsm = VectorFSM{SR}()
    for i in 1:3
        s1 = addstate!(ufsm, "a"; initweight = SR(0.5))
        s2 = addstate!(ufsm, "b"; initweight = SR(0.5))
        s3 = addstate!(ufsm, "c"; finalweight = SR(0.25))
        addarc!(ufsm, s1, s1, SR(0.75))
        addarc!(ufsm, s1, s2, SR(0.25))
        addarc!(ufsm, s2, s2, SR(0.75))
        addarc!(ufsm, s2, s3, SR(0.25))
        addarc!(ufsm, s3, s3, SR(0.75))
    end
    @test fsmequal(ufsm, union(mfsm, mfsm, mfsm))
end

