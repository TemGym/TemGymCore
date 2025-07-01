# Jax Tem Gym

A differentiable ray tracing platform that solves optical systems
via a Taylor Expansion of a "ray" representing the optical axis. 
Linear optics systems will return the ABCD values of a ray 
transfer matrix, and in the non-linear case will return coefficients of a polynomial, whose
values can be associated to aberration coefficients of the optics of the system. 

The returned polynomial equations representing an optical system can then be solved to determine output ray positions, 
slopes, amplitudes and phases, enabling one to propagate input wavefronts through linear or non-linear optical systems. 

The specific use case implemented in this library so far is designed to solve a linear system representing the coordinate transformation of a defocused point source on 
a sample, creating a "shadow image" of the sample on the detector. Shadow imaging can be realised in a Scanning Transmission Electron Microscope (STEM) experiment in an electron microscope, and it's extension to a number of scan positions entitled 4D STEM creates a dataset of shadow images on the detector as a function of scan position. 
Utilising the code in this repository, and a 4D STEM dataset of shadow images, we can solve the linear system of the shadow image projection, and by iteratively backprojecting each shadow image, can verify whether the coordinate system, and parameters such as scan step, camera length, scan rotation used in the 4D STEM experiment are correct. Such a verification step is neccessary in order to reliably apply iterative phase retrieval algorithms such as Ptychography to a 4D STEM experiment. 

The location of images on the detector in 4D STEM experiments can also suffer from a systematic error in the Scan/Descan system, which is used to raster the beam over the sample, and return it to the optical axis such that the beam is viewed on the centre of the detector. When shadow images are not returned to the centre of the detector, the STEM experiment suffers from Descan Error which must be corrected for before the coordinate system can be determined. We show how one can use a ray transfer matrix to represent Descan Error in the system, and how to fit it such that it can be corrected reliably for a range of camera lengths.

This work is a continuation of a non-differentiable ray tracing library designed for visualisation published in 2023
TemGym: Landers, D., Clancy, I., Weber, D., Dunin-Borkowski, R. & Stewart, A. (2023). J. Appl. Cryst. 56, https://doi.org/10.1107/S1600576723005174



