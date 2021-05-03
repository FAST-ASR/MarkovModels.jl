# Copyright - 2020 - Brno University of Technology
# Copyright - 2021 - CNRS
#
# Contact: Lucas Ondel <lucas.ondel@gmail.com>
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

const Label = Union{String,Nothing}
const PdfIndex = Union{Int, Nothing}

mutable struct State{T<:Semifield}
    id::Int
    pdfindex::PdfIndex
    label::Label
    startweight::T
    finalweight::T
end

#Base.:(==)(s1::State, s2::State) = s1.id == s2.id
#Base.hash(s::State, h::UInt) = hash(s.id, h)

isinit(s::State{T}) where T = s.startweight ≠ zero(T)
isfinal(s::State{T}) where T = s.finalweight ≠ zero(T)

setstart!(s::State{T}, weight::T = one(T)) where T = s.startweight = weight
setfinal!(s::State{T}, weight::T = one(T)) where T = s.finalweight = weight
