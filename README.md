# SPIKERR

PyTorch Library to facilitate **Spiking Neural Network research.**

## Goals

SPIKERR is a library for academics to iterate quickly on ideas. In order of importance:

1. [Simplicity](https://web.archive.org/web/20201001013648/http://www.catb.org/~esr/writings/taoup/html/ch13s01.html)
2. Generality
3. Speed

Developers are empowered to use their critical thinking skills on trade-offs. Trading a little simplicity may be worth a lot of speed.

SPIKERR should be considered an extension of [PyTorch](https://pytorch.org/). If a library works well with PyTorch, it should work well with SPIKERR.

SPIKERR's core competency is Spiking Neural Networks, abbreviated as SNNs. Auxiliary functionality should be delegated to well maintained 3rd party libraries that follow similar design philosophies. For example, PyTorch for tensor operations, matplotlib for graphing, etc. SPIKERR may provide helpful utilities around these libraries specifically for the domain of SNNs.

## Design Philosophy

SPIKERR follows PyTorch's powerful design patterns. SPIKERR has a functional core with auxiliary OOP wrappers. SPIKERR is functional oriented first, with classes as a higher "ease of use" tool.

SPIKERR is a layered library with simple "common use case" functions at the top, and more granular building blocks at the bottom.

### Major Principles

- Orthogonality: Do Not Link What Is Independent. A change in one orthogonal function has no observable effect on any other function in the set.
- Propriety: Do Not Introduce What Is Immaterial. The opposite of propriety is _extraneousness_. Shifting gears is not proper to driving; the extraneous component of the user interface arises from the implementation of the car.
- Generality: Do Not Restrict What Is Inherent. _Generality_ is the ability to use a function for many ends. It expresses the professional humility of the designer, his conviction that users will be inventitive beyond his imagination.

(Paraphrased from Fred Brook's Design of Design)

## Style Guide

SPIKERR follows the [Black code formatting style](https://black.readthedocs.io/en/stable/) and all the standard [PEP conventions](https://www.python.org/dev/peps/pep-0008/). Overrides may be added with strong justification. SPIKERR tools and guides are built around [PyCharm](https://www.jetbrains.com/pycharm/) as the common IDE, but you are welcome to use whichever editor you'd like.

## Contributing

All PRs, ideas, and fixes are welcome. If you are making major changes, a design doc before you start to get feedback may expedite the process.

Your PR must pass the automated tests. The [semver](https://semver.org/) should be incremented appropriately as well.

If you're looking for ideas, [the issues page](https://github.com/ELanning/SPIKERR/issues) is a good place to start.

## Onboarding

**TODO**
