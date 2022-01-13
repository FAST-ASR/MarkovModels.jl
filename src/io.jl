
import Base: write, read, isbitstype

write(io::IO, x::LogSemifield{T}) where T = write(io, x.val)
write(io::IO, cfsm::CompiledFSM{SF}) where SF <:Semifield = begin
    pdfmap = cfsm.pdfmap
    if  !(nothing in cfsm.pdfmap)
        pdfmap = convert(Vector{Int}, cfsm.pdfmap)
    else
        @error "We do not support pdfmap with elements of `Nothing` yet."
    end

    T = Float64
    if SF == LogSemifield{Float64}
        T = Float64
        write(io, "LSFD")
    elseif SF == LogSemifield{Float32}
        T = Float32
        write(io, "LSFF")
    else
        @error "Unsupported type $T"
    end

    A = convert(Matrix{T}, cfsm.A)
    N,M = size(A)
    write(io, UInt32(N), UInt32(M))
    write(io, reshape(A, (N*M,)))

    π = convert(Vector{T}, cfsm.π)
    write(io, UInt32(length(π)), π)

    ω = convert(Vector{T}, cfsm.ω)
    write(io, UInt32(length(ω)), ω)

    write(io, UInt32(length(pdfmap)), pdfmap)
end

function read(io::IO, ::Type{<:CompiledFSM})
    type = String(read(io, 4))
    T = nothing
    if type == "LSFF"
        T = Float32
        SF = LogSemifield{T}
    elseif type == "LSFD"
        T = Float64
        SF = LogSemifield{T}
    else
        @error "Unsupported type: $type"
    end

    N, M = Int(read(io, UInt32)), Int(read(io, UInt32))
    A = Array{T}(undef, N*M)
    read!(io, A)
    A = reshape(A, (N,M))
    Aᵀ = A'
    A = convert(Matrix{SF}, A)
    Aᵀ = convert(Matrix{SF}, Aᵀ)
    A = sparse(A)
    Aᵀ = sparse(Aᵀ)
    dropzeros!(A)
    dropzeros!(Aᵀ)

    N = Int(read(io, UInt32))
    π = Array{T}(undef, N)
    read!(io, π)
    π = convert(Vector{SF}, π)
    π = sparse(π)
    dropzeros!(π)

    N = Int(read(io, UInt32))
    ω = Array{T}(undef, N)
    read!(io, ω)
    ω = convert(Vector{SF}, ω)
    ω = sparse(ω)
    dropzeros!(ω)

    N = Int(read(io, UInt32))
    pdfmap = Array{Int}(undef, N)
    read!(io, pdfmap)
    pdfmap = convert(Vector{Int}, pdfmap)

    return CompiledFSM{SF}(π, ω, A, Aᵀ, pdfmap)
end
