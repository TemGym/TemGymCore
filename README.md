# Jax Tem Gym

A ray tracing package that uses the automatic differentiation tools of jax to solve optical systems
via a Taylor Expansion of a "ray" representing the optical axis. 
In TemGym, Linear optical systems are represented via the ABCD values of a ray 
transfer matrix determined using the Jacobian of the ray coordinates through the optical system, and non-linear optical systems are represented via 
coefficients of a Taylor Expansion polynomial of the ray coordinates through the optical system, calculated either via repeated Jacobian calls, 
or via the experimental library jax.jet.

The returned polynomial equations representing an optical system can then be solved to determine output ray positions, 
slopes, amplitudes and phases, enabling one to propagate input wavefronts through linear or non-linear optical systems. 

The specific use case implemented in this library is designed to solve a linear system representing the coordinate transformation of a defocused point source on 
a sample, creating a "shadow image" of the sample on the detector. Utilising the code in this repository, and a 4D STEM dataset of shadow images, we can solve the linear system of the shadow image projection, and by iteratively backprojecting each shadow image via ray tracing, can verify whether the coordinate system, and parameters such as scan step, camera length, scan rotation used in the 4D STEM experiment are correct. Such a verification step is neccessary in order to reliably apply iterative phase retrieval algorithms such as Ptychography to a 4D STEM experiment. 

The location of images on the detector in 4D STEM experiments can also suffer from a systematic error in the Scan/Descan system, which is used to raster the beam over the sample, and return it to the optical axis such that the beam is viewed on the centre of the detector. When shadow images are not returned to the centre of the detector, the STEM experiment suffers from Descan Error which must be corrected for before the coordinate system can be determined. We show how one can use a ray transfer matrix to represent Descan Error in the system, and how to fit it such that it can be corrected reliably for a range of camera lengths.

This work is a continuation of a non-differentiable ray tracing library designed for visualisation published in 2023
TemGym: Landers, D., Clancy, I., Weber, D., Dunin-Borkowski, R. & Stewart, A. (2023). J. Appl. Cryst. 56, https://doi.org/10.1107/S1600576723005174



