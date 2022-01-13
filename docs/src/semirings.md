# Semirings

Algorithms implemented in this package are very generic as they can be
applied on various type of "numbers":
  * probabilities (values between ``0`` and ``+\infty``)
  * log-probabilities (values between ``-\infty`` and ``+\infty``)
  * ...
Formally, they can operate on any [semiring](https://en.wikipedia.org/wiki/Semiring)
or [semifield](https://en.wikipedia.org/wiki/Semifield) for some of them.

To work with these algebraic structures, the package defines the
following abstract types:

```@docs
Semiring
Semifield
```

## Concrete types

The package provides the following concrete types:

```@docs
LogSemifield
ProbabilitySemifield
TropicalSemiring
```

