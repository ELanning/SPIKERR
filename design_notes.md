# Design Notes

Documented here are the finer design decisions, with reasoning behind each concept.

* Do design all APIs around **2D tensors**.

This lends itself to consistency and simplicity across the project. Consumers wanting to use more dimensions can still construct an isomorphism by adding more layers, or other methods.
Supporting more dimensions increases degenerate edge cases and causes confusion. This means a set of voltages, weights, etc should be 1xN NOT N.

* Do use abstract base classes if it reduces complexity.

Abstract base classes are a powerful tool for code reuse and maintaining conceptual integrity. However.

* Do NOT use inheritance hierarchies for code reuse.

Much like `GOTO`, most research and compiled experience has shown inheritance hierarchies to be brittle, confusing, and not worth the trouble. Inheriting from **ONE** abstract base class is the limit, unless you can prove your case.

* Do add `Type` annotations to every public interface.

Type annotations are documentation and free code checking. They communicate intent and make code more maintainable.

* Do NOT add long and verbose JDocs to every API.

Much of this API is _experimental_. Do not bother writing long prose on something that may be deleted the next day. Having to write long JSDocs on everything kills project momentum and motivation.

* Do add unit tests and/or integration tests to public APIs.

Tests communicate requirements, document how to use an API, catch regressions, and catch bugs. Any API without a minimum of one test cannot be considered public.

* Do document expectations by throwing exceptions if they are not met.

However, do not go overboard. This is largely a matter of taste. This project follows the [FailFast](https://wiki.c2.com/?FailFast) methodology.

* Do NOT add multiple ways to do one thing.

If a convenience function would greatly improve the dev experience, delete the old way instead of adding both. This does not mean you cannot provide a high level PyTorch class that wraps lower level functions. It means no `LeakyIntegrateAndFire1` and `LeakyIntegrateAndFire2`.

* Do NOT [couple](https://wiki.c2.com/?CouplingAndCohesion) modular units together without good reason.

Instead prefer passing `Callable` functions. A button does not know about all the different ways it is used, instead it has an `onclick` function that lets consumers plugin whatever is needed. SPIKERR should follow a similar philosophy. However, this is a matter of taste and best judgement.

* Do use common **DOMAIN** abbreviations.

In most code, abbreviations are best avoided. However, SPIKERR is built around the domain of spiking neural networks, thus common abbreviations leverage shared knowledge. Scientific and mathematical names often have the tendency to be extremely long. Thus, feel free to use common abbreviations, but comment what the abbreviation stands for at least once.

* Do NOT use nonsense abbreviations. `count` is not improved by changing it to `cnt`.