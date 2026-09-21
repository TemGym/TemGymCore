# Seidel action on a Gaussian

`SeidelLens` and `DistortedLens` accept an incoming Gaussian, the lens geometry,
and aberration coefficients. The reference object coordinate is inferred as
$o_0=a_0-z_1\theta_0$; it is an internal integration variable, not a required
user input. The three Seidel parameterizations describe the same polynomial,
and their gradients and Hessians agree after the coordinate transformation.

## Optical convention

The object-to-pupil kernel is

$$
K(a,o)=\frac{k}{2\pi i z_1}
\exp\!\left[ik\left(\frac{|a-o|^2}{2z_1}+s(o,a)\right)\right],
\qquad s(o,a)=-\frac{|a-a_L|^2}{2f}-W(o,a).
$$

The reference object field is the inverse Fresnel propagation of the incoming
pupil field. For $W=0$, this kernel therefore gives the ordinary thin lens.
A prescribed `object_xy` retains the point-object pupil phase convention.

The kernel amplitude is the existing Fresnel amplitude. The polynomial $W$
specifies wave aberration; field-dependent pupil apodization is not specified.
Consequently this model does not impose flux conservation for arbitrary
field-dependent coefficients. For pure distortion, the unbounded-pupil integral
gives the usual cubic image displacement with a field factor $1/|\det DF|$.
The notebook uses that integral as a reference, without fitting its normalization.

## Local expansion

First locate the central ray at the pupil. Differentiating the total optical
path with respect to the object coordinate gives

$$
a_* - a_0-z_1 s_o(o_0,a_*)=0.
$$

The implementation uses six Newton steps from $a_0$, selecting the nearby
smooth branch. For pure distortion this equation is linear in $a_*$.
Expand $s$ jointly about $(o_0,a_*)$, then translate the resulting quadratic
polynomial to $(o_0,a_0)$. This translation preserves that polynomial exactly;
expanding initially at $a_0$ can miss the wavefront around a displaced ray.

Let the translated value, gradient, and Hessian be $s_0$, $(g_o,g_a)$, and
$H_{oo},H_{oa},H_{ao},H_{aa}$. The object-plane Gaussian has curvature $Q_o$,
where $Q$ denotes the code's `Q_inv`. Define

$$
G=Q_o+I/z_1+H_{oo},\qquad K=-I/z_1+H_{oa}.
$$

Completing the square in the object-coordinate integral gives

$$
\Delta S_0=s_0-\tfrac12g_o^T G^{-1}g_o-\frac{i}{k}\log A,
$$
$$
\Delta S_1=g_a-K^T G^{-1}g_o,
\qquad
\Delta S_2=I/z_1+H_{aa}-K^T G^{-1}K-Q,
$$

where

$$
A=\frac{\sqrt{\det(I+z_1 Q_o)}}
        {\sqrt{\det[I+z_1(Q_o+H_{oo})]}}.
$$

The square roots follow the same continuous Fresnel branch as free-space
Gaussian propagation. These are complex **spatial** action coefficients and
are passed directly to the existing `apply_action_delta`. The implementation
uses equivalent expressions without division by $z_1$, including its zero limit.
A pupil-only action reduces to the existing spatial Taylor update.

## Checks and limits

The quadratic reduction is checked against direct two-dimensional diffraction
quadrature for positive and negative propagation distances. Further checks
cover zero aberration, pupil-only spherical aberration, equivalent Seidel
parameterizations, batched cubic image displacement, and coefficient gradients.
The notebook compares the full distortion polynomial with direct quadrature,
and propagates 220,000–330,000 coherent Gaussians through PL1.

The quadratic integration is exact for its quadratic kernel; the Seidel
polynomial is only locally approximated. The source and pupil widths must
resolve the omitted terms, and the central-ray solve must remain on a regular
branch. Increasing the beam count alone does not establish this: the notebook
checks complex fields and changes source widths as well as sampling density.
