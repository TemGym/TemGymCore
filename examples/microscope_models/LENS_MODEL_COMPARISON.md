# The two finite lens models

The reduction experiment keeps the hard-edge solenoid used to generate the
images. This note compares it with TemGym's existing `ElectromagneticLens` in
[`components.py`](../../src/temgym_core/components.py).

## The older electromagnetic component

With the stigmator disabled, it applies half of the focusing power, propagates
a distance $t$, applies the other half, and rotates by $\phi$. In one transverse
direction its matrix before rotation is

$$
S_{\mathrm{split}}=
\begin{pmatrix}
1-t/(2f) & t \\
-1/f+t/(4f^2) & 1-t/(2f)
\end{pmatrix}.
$$

The full matrix in $(x,y,\theta_x,\theta_y,1)$ coordinates is

$$
M=\begin{pmatrix}
aR(\phi)&bR(\phi)&\mathbf 0\\
cR(\phi)&dR(\phi)&\mathbf 0\\
0&0&1
\end{pmatrix},\qquad S=\begin{pmatrix}a&b\\c&d\end{pmatrix}.
$$

The component's actual control laws are

$$
f=\frac{V^*}{G_c(NI)^2},\qquad
t=T_c|NI|,\qquad
\phi=R_c(V)NI.
$$

Here $NI$ is turns times current. The current implementation computes $R_c$
from voltage; it is not a free per-lens attribute. Allowing independent
calibrations for $f$, $t$ and $\phi$ would provide three response parameters at
one excitation. This is more freedom than the two parameters below.

The split map has determinant one. Its `focal_length` input sets the combined
power of the two kicks. At nonzero thickness its equivalent focal length is
$f_{\mathrm{eff}}=f/(1-t/(4f))$, from $-1/c$. Those two focal lengths should not
be confused during fitting.

There is also a separate `ThickLens` class, parameterized by two principal-plane
positions and focal length. It applies a thin-lens kick and changes the ray's
axial coordinate; it has no rotation parameter and no Gaussian-beam method.

## The solenoid used in these notebooks

For a uniform field of fixed length $L$, integrated strength $q$ gives

$$
S_{\mathrm{solenoid}}=
\begin{pmatrix}
\cos q & L\,\sin q/q\\
-(q/L)\sin q & \cos q
\end{pmatrix},\qquad \phi=-q.
$$

It uses the same grouped block layout. Focusing and rotation come from the same
field strength. Its exact paraxial equivalent focal length is
$L/(q\sin q)$; $L/q^2$ is the weak-lens approximation. Lens position is an
additional fitted parameter in either column model.

## Which is useful here?

The older component has readable controls and can vary focusing independently
of rotation and thickness. That freedom may help an effective model reproduce
responses that a uniform field cannot. It also assumes thickness grows with
excitation, and the split map is not the exact uniform-field propagation map.
More freedom alone does not establish greater physical accuracy.

The solenoid has a smaller, physically linked parameter set and fixed axial
extent. It is a clean choice for the current controlled recovery experiment.
Both models need validation before interpreting fitted dimensions as mechanical
lens dimensions. The dataset-reduction result applies to the solenoid's 19
parameters; it does not establish identifiability of the older model's extra
freedom. A comparison of model families would be a separate experiment.
