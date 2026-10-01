"""Pair-loop implementations of the renderer primitives.

``reference`` is pure JAX (the CPU implementation and the specification of the others),
``dense`` and ``sparse`` are the Pallas/Triton float64 kernels for CUDA, and ``common`` holds what
they share.
"""
