# SPDX-License-Identifier: MIT


abstract type FSAProperty end


struct Accessible <: FSAProperty end
struct Acyclic <: FSAProperty end
struct CoAccessible <: FSAProperty end
struct Deterministic <: FSAProperty end
struct LexicographicallySorted <: FSAProperty end
struct Normalized <: FSAProperty end
struct TopologicallySorted <: FSAProperty end
struct WeightPropagated  <: FSAProperty end


const accessible = Accessible()
const acyclic = Acyclic()
const coaccessible = CoAccessible()
const deterministic = Deterministic()
const lexsorted = LexicographicallySorted()
const normalized = Normalized()
const topsorted = TopologicallySorted()
const weightpropagated = WeightPropagated()


struct Properties{
        Taccessible <: Union{Accessible, Missing},
        Tacyclic <: Union{Acyclic, Missing},
        Tcoaccessible <: Union{CoAccessible, Missing},
        Tdeterministic <: Union{Deterministic, Missing},
        Tlexsorted <: Union{LexicographicallySorted, Missing},
        Tnormalized <: Union{Normalized, Missing},
        Ttopsorted <: Union{TopologicallySorted, Missing},
        Tweightpropagated <: Union{WeightPropagated, Missing},
    }

    accessible::Taccessible
    acyclic::Tacyclic
    coaccessible::Tcoaccessible
    deterministic::Tdeterministic
    lexsorted::Tlexsorted
    normalized::Tnormalized
    topsorted::Ttopsorted
    weightpropagated::Tweightpropagated
end


Properties(;
    accessible = missing,
    acyclic = missing,
    coaccessible = missing,
    deterministic = missing,
    lexsorted = missing,
    normalized = missing,
    topsorted = missing,
    weightpropagated = missing
) = Properties(
    accessible,
    acyclic,
    coaccessible,
    deterministic,
    lexsorted,
    normalized,
    topsorted,
    weightpropagated
)


const HasAccessibleProperty = Properties{
        Accessible,
        <:Union{Acyclic, Missing},
        <:Union{CoAccessible, Missing},
        <:Union{Deterministic, Missing},
        <:Union{LexicographicallySorted, Missing},
        <:Union{Normalized, Missing},
        <:Union{TopologicallySorted, Missing},
        <:Union{WeightPropagated, Missing},
    }

const HasAcyclicProperty = Properties{
        <:Union{Accessible, Missing},
        Acyclic,
        <:Union{CoAccessible, Missing},
        <:Union{Deterministic, Missing},
        <:Union{LexicographicallySorted, Missing},
        <:Union{Normalized, Missing},
        <:Union{TopologicallySorted, Missing},
        <:Union{WeightPropagated, Missing},
    }

const HasCoAccessibleProperty = Properties{
        <:Union{Accessible, Missing},
        <:Union{Acyclic, Missing},
        CoAccessible,
        <:Union{Deterministic, Missing},
        <:Union{LexicographicallySorted, Missing},
        <:Union{Normalized, Missing},
        <:Union{TopologicallySorted, Missing},
        <:Union{WeightPropagated, Missing},
    }

const HasDeterministicProperty = Properties{
        <:Union{Accessible, Missing},
        <:Union{Acyclic, Missing},
        <:Union{CoAccessible, Missing},
        Deterministic,
        <:Union{LexicographicallySorted, Missing},
        <:Union{Normalized, Missing},
        <:Union{TopologicallySorted, Missing},
        <:Union{WeightPropagated, Missing},
    }

const HasLexicographicallySortedProperty = Properties{
        <:Union{Accessible, Missing},
        <:Union{Acyclic, Missing},
        <:Union{CoAccessible, Missing},
        <:Union{Deterministic, Missing},
        LexicographicallySorted,
        <:Union{Normalized, Missing},
        <:Union{TopologicallySorted, Missing},
        <:Union{WeightPropagated, Missing},
    }

const HasNormalizedProperty = Properties{
        <:Union{Accessible, Missing},
        <:Union{Acyclic, Missing},
        <:Union{CoAccessible, Missing},
        <:Union{Deterministic, Missing},
        <:Union{LexicographicallySorted, Missing},
        Normalized,
        <:Union{TopologicallySorted, Missing},
        <:Union{WeightPropagated, Missing},
    }

const HasTopologicallySortedProperty = Properties{
        <:Union{Accessible, Missing},
        <:Union{Acyclic, Missing},
        <:Union{CoAccessible, Missing},
        <:Union{Deterministic, Missing},
        <:Union{LexicographicallySorted, Missing},
        <:Union{Normalized, Missing},
        TopologicallySorted,
        <:Union{WeightPropagated, Missing},
    }

const HasWeightPropagatedProperty = Properties{
        <:Union{Accessible, Missing},
        <:Union{Acyclic, Missing},
        <:Union{CoAccessible, Missing},
        <:Union{Deterministic, Missing},
        <:Union{LexicographicallySorted, Missing},
        <:Union{Normalized, Missing},
        <:Union{TopologicallySorted, Missing},
        WeightPropagated
    }

