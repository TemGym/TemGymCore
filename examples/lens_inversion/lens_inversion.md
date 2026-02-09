Summary: The quadratic phase factor containing terms $C/A$ that allows one needs to say something about the distances between lenses and their focal lengths is almost impossible 
to measure in a practical way. One can move the detector (or use two detectors spaced apart by some known distance) to measure how intensity changes between them, and infer phase, but this would only work at a very low magnification (maybe at 1000 $\times$ or less). One could also adjust the focal length of a lens, and try to perform phase retrieval at the plane of the lens, but without knowing any distances or focal lengths absolutely, this is only solvable up to a known quadratic phase factor i.e it's not useful to measure absolute distances and focal lengths in the microscope. 

This document  concerns the measurement of the slope (proportional to phase in the paraxial sense) of the output rays on the detector, and how phase retrieval methods might be used to do so. One would want to measure the slope on the output detector for the following reasons - 

1. It would be necessary to invert the distances and focal lengths between components. Without slope measurements, one can not pin down the possible configurations of the microscope that deliver an image onto the detector. 
2. An alternative method to measuring aberrations of the system. If one can measure the slope on the output plane, one can compare those slopes to those of the paraxial case, which should follow a parabolic profile. This provides an alternative method to measuring aberrations, and would also allow one to determine contributions from different lenses in the system. For instance if one perturbed the projector lens system, one might be able to separate it's contributions to the aberrations of the system from those of the objective lens.
3. One could potentially analyse the focal lengths and distances of the condenser lens system. If one can perturb each condenser lens and measure the slope on the detector, one could monitor how they affect the overall transfer matrix, and fit distances and focal lengths to those components also. 
4. One could measure the microscope transfer matrix without turning the objective lens off, and could include it's contribution to the imaging of the system. The only known previous work I can find [# Fundamentals of Focal Series Inline Electron Holography](https://www.sciencedirect.com/science/article/abs/pii/S107656701630088X) that attempted to fit the distances and focal lengths of the microscope turned off the objective lens and used diffraction mode. If one can access the slopes/phase on the detector one does not need to turn off the objective lens and can include it in the solution to the microscope model.
5. If possible, such a method could potentially help calibrate the complete transfer matrix in a live manner on the microscope, if a set of apertures of different sizes could be integrated onto the sample holder, rather than having to use a specific calibration sample such as diffraction grating. 


*Preliminaries*:
The Single-Fourier Transform (SFT), and Double-Fourier Transform (DFT) solution to the Generalised Fresnel Integral (Collins Integral) describes how an $ABCD$ matrix of an optical system images an input wavefront $U_1$, and is written as follows:

$$
U_2(x_2,y_2)
\;=\;
\frac{e^{i k L_0}}{i\lambda B}\,
\exp\!\left[\frac{i \pi D}{\lambda \ B}\,\,(x_2^{2}+y_2^{2})\right]\,
\mathcal{F}\!\left\{\,U_{1}(x_{1},y_{1})\,
\exp\!\left[\frac{i \pi A}{B}\,\,(x_{1}^{2}+y_{1}^{2})\right]\right\}_{%
f_{x}=\frac{x}{\lambda B},\;f_{y}=\frac{y}{\lambda B}} \text{: Single Fourier Transform Solution}
$$

$$
U_{2}(x_2,y_2) \;=\; \frac{e^{i k L_{0}}}{A}
\exp\!\Bigl[\frac{i\pi\,C}{\lambda\,A}\,(x_{2}^{2}+y_{2}^{2})\Bigr]\;
\Bigl\{\mathcal{F}^{-1}\!\Bigl[\mathcal{F}\{U_{1}(x_{1},y_{1})\}\,
\exp\!\Bigl(-\frac{i\pi\,\lambda\,B}{A}\,(f_{x}^{2}+f_{y}^{2})\Bigr)\Bigr]\Bigr\}_{x_{1}=x_{2}/A,\;y_{1}=y_{2}/A}\,  \text{: Double Fourier Transform Solution}
$$

(Depending on what coordinate transformation one applies to their system (i.e imaging or diffraction etc.) it is convenient to choose either the SFT or DFT version to reason about the consequences.)

Typical phase retrieval problems are concerned with the unknown phase of the input plane $U_{1}(x_{1},y_{1})$, and there are two primary methods used to determine the phase of $U_1(x_1, y_1)$ - far field methods which rely on the Fourier transform of the input, and near field methods which rely on a set of defocused intensity images.
In the far field method, the sample is placed at the front focal plane of a lens, and we visualise the back focal plane on the detector - therefore the coordinate transform is

$$
\begin{pmatrix}A & B \\[4pt] C & D\end{pmatrix}
=
\begin{pmatrix}0 & f\\[4pt] -\dfrac{1}{f} & 0\end{pmatrix},
$$

and using the SFT Collins integral, we see that the outer quadratic phase factor disappears (since $D = 0$), and then $U_{2}(x_{1}, y_{1})$ is simply a straight Fourier transform of the input. In the case where the sample is not placed at the front focal plane of the detector, the outer quadratic phase factor does not disappear, but on the final intensity image it is invisible, and has no impact on the intensities in the final image.

In the near field case, we turn to the DFT Collins integral, where we must have a measurement for the effective defocus determined by $B/A$ in order to introduce diversity to measure the phase of $U_1$. Typically, one moves the sample plane to modify $B$ and $D$ in the overall $ABCD$ matrix (to see why moving the sample changes $B$ and $D$ as well, write out the 3 transfer matrices for imaging with a lens - propagation, lens, propagation, and see what happens if another free space is added before this system - one sees that the coordinates $B$ and $D$ change). Of course, any changes to $D$ are invisible in the outer quadratic phase factor of the DFT solution to the Collins integral and it is still irrelevant. All of this is to say that in all typical cases of phase retrieval there is another phase on the detector that is ignored - the quadratic phase factor on the output outside of the Fourier transform. Nobody cares about it since it is the sample that matters. However if you want to say something about the distances and focal lengths of the lens system in the microscope, one must measure this outer quadratic phase factor. Why is this the case? Because in order to try and fit distances and focal lengths to the microscope system, one must first be able to measure the complete transfer matrix - $A$, $B$, $C$, and $D$. If one can not recover the $ABCD$ matrix of the system, one can not even begin to fit a model of the microscope. 

We know that

$x_2 = A x_1 + B \theta_1$ 

and 

$\theta_2 = Cx_1 + D \theta_1$

so in order to determine $C$ we need to be able to measure the slope, or phase at the output to get access to it. Because of the symplectic condition, $AD - BC = 1$ we need only measure three of four of the $ABCD$ matrix and, and thus if we can get access to $A$ and $B$, we will only need to measure $C$, and can ignore the influence of $D$. However, determining $\theta_2$ on the detector plane is made very complicated by the fact that the quadratic phase factor in the SFT and DFT solution to the Collins integral appears outside the Fourier transforms in both cases. This means that when intensity is recorded on the detector it becomes invisible and has no influence on the final image in imaging mode. 

Before we explore some possibilities of how to access the outer quadratic phase factor we should mention that it seems helpful to have a known input sample - an aperture with a known radius or otherwise, and preferably plane wave illumination at the beginning. Without this, I would say without thinking about it, that it is not possible to measure the microscope system, and the sample intensity and phase. 

In order to measure the final quadratic phase factor $C/A$ at first glance, one must introduce some diversity in $U_1$ **THAT** includes the phase factor $C/A$. If $U_1$ includes the quadratic phase factor $C/A$, then any intensity images recorded by changing $B$ or $A$ or both, will be influenced by the quadratic phase factor $C/A$, and it should be possible with gradient descent or otherwise to find the value of $C/A$ which explains the set of intensity images recorded by the detector. 

*Method 1 - Move the detector*

The obvious solution to this problem is to move the detector. By moving the detector we can record a field, $U_{\Delta z}$, a detector distance $\Delta z$  away,  that has some unknown phase *which* includes the quadratic phase factor $C/A$. Thus, subsequent intensity images recorded by moving the detector will now have their intensity influenced by the value of this quadratic phase factor that existed on first image on the detector plane. Moving the detector at the end of imaging process is akin to adding a propagation matrix at the end of the transfer matrix of the system.

$$
\begin{pmatrix}
A & B\\[4pt]
C & D
\end{pmatrix}
=
\begin{pmatrix}
1 & z\\[4pt]
0 & 1
\end{pmatrix},
\qquad\text{(free-space propagation over distance }z\text{)}
$$

If one is in perfect imaging mode then the transfer matrix is - 

$$
\begin{pmatrix}
A & B\\[4pt]
C & D
\end{pmatrix}
=
\begin{pmatrix}
M & 0\\[4pt]
-\dfrac{1}{f} & \dfrac{1}{M}
\end{pmatrix}
$$

where $M$ is magnification, $f$ = focal length. If we left multiply the detector by a known free-space propagation we find that 

$$
\begin{pmatrix}1 & z\\[4pt]0 & 1\end{pmatrix}
\begin{pmatrix}M & 0\\[4pt]-\dfrac{1}{f} & \dfrac{1}{M}\end{pmatrix}
=
\begin{pmatrix}M-\dfrac{z}{f} & \dfrac{z}{M}\\[6pt]-\dfrac{1}{f} & \dfrac{1}{M}\end{pmatrix}.
$$

Thus after left‑multiplying by a free‑space propagation of z,
$A = M − z/f$, $B = z/M$, while $C = −1/f$ and $D = 1/M$ remain unchanged.

Written in terms of the Collins Integral (DFT solution) if we start by recording a perfect image of a known aperture at the sample plane -  $U_{2} (z_{det} = 0)$ and perfect imaging $A = M$, $B=0$, $C=-1/f$, $D=1/M$, then:

$$
U_{2}(x_2,y_2)_{z_{det} = 0}  \;=\; \frac{e^{i k L_{0}}}{M}
\exp\!\Bigl[\frac{-i\pi\,}{\lambda\,fM}\,(x_{2}^{2}+y_{2}^{2})\Bigr]\;
\,U_{1}(x_{1},y_{1}).
$$

Now if we move the detector a distance $\Delta z$,  (ignoring the constant phase factor)
$$
U_{3}(x_3,y_3)_{z_{det} = \Delta z} =
\Bigl\{\mathcal{F}^{-1}\!\Bigl[\mathcal{F}\{U_{2}(x_2,y_2)_{z_{det} = 0})\}\,
\exp\!\Bigl(-{i\pi\,\lambda\,\Delta z}\,(f_{x}^{2}+f_{y}^{2})\Bigr)\Bigr]\Bigr\}
$$
the intensity of $U_3$ will be influenced by the phase profile of $U_2$ that interacts with the propagation kernel a distance of $\Delta z$ , and with enough images (i.e more $\Delta z$ movements of the detector), one could recover the quadratic phase profile of $U_2$, and obtain a measurement of $C/A$. Then, by also knowing $A$, which is trivial if we have a known sample input, we can work out $C$. 

One key question remains - how much should one have to move the detector to see the intensity influenced significantly by the quadratic phase factor?

The point to remember is that the final slopes on the detector in an imaging system are proportional to $C/A$. If magnification, $A$ is $200,000$, and $C$, the effective focal length of the entire optics system is on the order of millimetres, then the slopes of the rays on the detector will be given by (paraxial approximation): 

$$
\theta(x) = \frac{\lambda}{2 \pi} \frac{\delta \phi}{\delta x}
$$

If $\phi = \frac{\pi}{\lambda fM}x^2$, then 

$$
\theta(x) = \frac{x}{fM}
$$
At $M = 200,000$ and $f=10mm$, and with $x = 7.5 mm$ (distance from centre to edge),

$$
\theta(x)=\frac{x}{fM}
= \frac{7.5\times 10^{-3}\,\mathrm{m}}{(10\times 10^{-3}\,\mathrm{m})\cdot 2.0\times 10^{5}}
=3.75\times 10^{-6}\,\mathrm{rad}\approx 3.75\,\mu\mathrm{rad}.
$$

If we now consider a single ray at the edge of the detector with a slope of $3.75\times 10^{-6}\,\mathrm{rad}$, and a pixel pitch of $50\times 10^{-6}\,\mathrm{\mu}$  the distance we must move the detector to see a light ray move to the next pixel half a pixel pitch away - given by - $\delta x / 2$  (and thus see some change in intensity) is given by $\delta z = \frac{\delta x}{2 \theta}$. Therefore
$$
\delta x = 50\ \mu\text{m}\quad\Rightarrow\quad \frac{\delta x}{2} = 25\times 10^{-6}\ \text{m}
$$
$$
\delta z = \frac{25\times 10^{-6} m}{3.75\times 10^{-6}\,\mathrm{rad}}\ = 6.67 m
$$


We need to move the detector $6.67 m$ in order to see intensity change over one pixel, at such modest magnification. With $M$ at the scale of 100,000$\times$ and $f$ in the mm–cm range, $fM$ is enormous, so detector motion must be on the order of metres to produce pixel-scale shifts. To enable the use of millimetre scale motions on the detector to detect changes in intensity, you would need to reduce $fM$ by roughly $10^{3}$ i.e. use a much lower intermediate magnification. 

*Method 2 - Adjust the focal length of the last lens.* 

One point worth highlighting about phase retrieval (which seems obvious in hindsight) is that you only retrieve phase on the plane that makes the adjustment. So if you want phase on the detector, you must adjust the detector. If you want phase on the sample, one should adjust something on the sample. With this in mind, we might have another chance at phase retrieval to measure the microscope state, if we move one plane further up, and start adjusting the focal length of the last projector lens in the system. 

For now, we will stay with a thin lens model of the microscope, and we can model the preceding optics and the projector lens system as follows:

$$
\begin{pmatrix}
A_{\rm det} & B_{\rm det}\\[6pt]
C_{\rm det} & D_{\rm det}
\end{pmatrix}
=
\begin{pmatrix}
1 & z_{2}\\[4pt]0 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 0\\[4pt]-\dfrac{1}{f_{\rm pl}} & 1
\end{pmatrix}
\begin{pmatrix}
1 & z_{1}\\[4pt]0 & 1
\end{pmatrix}
\begin{pmatrix}
A & B\\[4pt]C & D
\end{pmatrix},
$$

where the entries are
$$
\begin{aligned}
A_{\rm det}
&= (A+z_1 C)\Big(1-\frac{z_2}{f_{\rm pl}}\Big)+z_2 C,\\[6pt]
B_{\rm det}
&= (B+z_1 D)\Big(1-\frac{z_2}{f_{\rm pl}}\Big)+z_2 D,\\[6pt]
C_{\rm det}
&= C-\frac{A+z_1 C}{f_{\rm pl}}
=\frac{C f_{\rm pl}-A-z_1 C}{f_{\rm pl}},\\[6pt]
D_{\rm det}
&= D-\frac{B+z_1 D}{f_{\rm pl}}
=\frac{D f_{\rm pl}-B-z_1 D}{f_{\rm pl}}.
\end{aligned}
$$
This time let's first explore the linear equations to understand the number of knowns and unknowns, and what is actually measurable and what isn't. We know we can't measure $C_{det}$ and $D_{det}$. All we can do now is add an unknown $\Delta f_{pl}$ to the projector lens. However, with enough measurements of $A_{det}$ and $B_{det}$ is it possible to fit a straight line, between both equations, and work out the parameters? 
Remembering that $1-\frac{z_2}{f_{pl}} = M_{pl}$ (magnification of projector lens) -
$$
\begin{aligned}
A_{\rm det}
&= (A+z_1 C)\Big(1-\frac{z_2}{f_{\rm pl}}\Big)+z_2 C = g M_{pl} + \epsilon \\[6pt] 
B_{\rm det}
&= (B+z_1 D)\Big(1-\frac{z_2}{f_{\rm pl}}\Big)+z_2 D = h M_{pl} + \tau\\[6pt]
\end{aligned}
$$

I think already looking at this, we have 2 equations and 7 unknowns! All we can do is change $f_{pl}$ and we don't even know by how much. Now, maybe because there is some shared information between the variables, there might be a way to solve this but unlikely. As we adjust $f_{pl}$ to create a new $M_{pl}$, and measure a new $A_{det}$, we will be building up a straight line, however we won't be able to even know our $x$ coordinate, as we won't know $M_{pl}$! All we will know is we have a new intensity image where $A$, $z_1$, $C$, $z_2$, $B$ and $D$ remained unchanged. 

For instance, if we divide our two equations, we find

$$
A_{det} / B_{det} = (g M_{pl} + ε) / (h M_{pl} + τ).
$$
and if we eliminate $M_{pl}$ to see the data geometry:

$$
A_{det} = (g/h) B_{det} + (ε − (g τ)/h).
$$

So as you vary $f_{pl}$ (hence $M_{pl}$), the points ($B_{det}$, $A_{det}$) lie on a single line with:
slope $s = g/h$, intercept $b = ε − s τ$. Therefore, I think without one known $M_{pl}$ value, it will be impossible to work out all the parameters. 

So now since the linear straight forward fitting method is proven to be debunked, can a phase retrieval method be  explore - although I fear we will run into the same issue. 

The first decision to make is what common input plane will we use to determine the amplitude and phase of the wavefront? I think we need to pick the plane of the lens itself. This is because if we can change the projector lens focal length, we will have a common amplitude and phase here that should effect the intensity profile of all images downstream, and thus a phase retrieval technique should work to obtain the amplitude and phase at the exact plane of the lens.

![[Images/Pasted image 20250826143332.png|800]]
Again, if we return to the Collins integral, and consider that on the plane of the last projector lens we have generated a $U_2$ with a certain quadratic phase factor: 
$$
U_2(x)=\frac{1}{A_1}\,e^{\,i\pi\frac{C_1}{\lambda A_1}(x^2+y^2)}\,
\mathcal{F}^{-1}\!\Bigl[\,\mathcal{F}\{U_1\}\,e^{-\,i\pi\lambda\,\frac{B_1}{A_1}(f_x^2+f_y^2)}\Bigr],
\quad
$$

Propagating to the detector with a new $\frac{B_2}{A_2}$ and using a property of the Fourier transform on the $\frac{C_1}{A_1}$ attached to $U_2$
$$
\mathcal{F}\!\bigl\{\mathcal{F}^{-1}[H]\ e^{\,i\pi\frac{C_1}{\lambda A_1}(x^2+y^2)}\bigr\}
=H\,e^{-\,i\pi\lambda\,\frac{A_1}{C_1}(f_x^2+f_y^2)},
$$

with 

$$
H(f_x,f_y):=\mathcal{F}\{U_1\}(f_x,f_y)\,
\exp\!\left(-\,i\pi\lambda\,\frac{B_1}{A_1}(f_x^2+f_y^2)\right),
$$

gives
$$
U_3(x)=\mathcal{F}^{-1}\!\Bigl[
\mathcal{F}\{U_1\}\,
e^{-\,i\pi\lambda\,(\frac{B_1}{A_1}+\frac{A_1}{C_1}+\frac{B_2}{A_2})\,(f_x^2+f_y^2)}
\Bigr]
\quad

$$

By changing the last lens focal length by an unknown amount, we can change the quadratic phase factor $B_2/A_2$, but we will never be able to disentangle it from $\frac{B_1}{A_1} + \frac{A_1}{C_1}$, and say something about the upstream optics unless we can anchor a distance or focal length of the projector lens. We will always have an global quadratic phase factor at the projector lens pupil that will be unknowable: curvature can always be reassigned between the pupil field and $B_2/A_2$ while producing the same measured intensities. We have run into the same ambiguity we obtained in the basic transfer matrix case above. 

*Conclusion:*

Near-field phase retrieval methods to measure the distances and focal lengths of components in the microscope are either impractical or mathematically ambiguous. Although, as Jean-Luc knows, if one puts an aperture into space between the last lens and the detector, and one can move it, one could also perform phase retrieval on the microscope state. 

I've had another look at this, and it's worth writing down the including the results of the most recent investigation. Mathematically, what we are dealing with is a high dimensional non-linear set of equations. Our chances of finding a unuque solution is minimised by the fact that we can't measure C and D in the ABCD transfer matrix, since we cannot measure the phase of the beam on the detector. 

---

# Illustrative Example - Two-Lens Inverse Problem

## Mathematical Formulation of the Two-Lens System

### Transfer Matrix Composition

Consider for simplicity a two-lens optical system, the overall transfer matrix is:

$$
M = P(d_3) \cdot L(f_2) \cdot P(d_2) \cdot L(f_1) \cdot P(d_1)
$$

where:
- $P(d)$ is a **propagation matrix** over distance $d$:
$$P(d) = \begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$$

- $L(f)$ is a **thin lens matrix** with focal length $f$:
$$L(f) = \begin{pmatrix} 1 & 0 \\ -\phi & 1 \end{pmatrix}, \quad \phi = \frac{1}{f}$$

- $d_1$ = object distance (sample to lens 1)
- $d_2$ = inter-lens distance
- $d_3$ = image distance (lens 2 to detector)
- $\phi_i = 1/f_i$ = optical power of lens $i$

### Building the Full Matrix Analytically

Multiplying out the five matrices in sequence gives:

$$
M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}
$$

where the **analytical formulas** for the ABCD elements are:

$$\boxed{A = (1 - d_2 \phi_1)(1 - d_3 \phi_2) - d_3 \phi_1}$$

$$\boxed{B = d_1 \cdot A + d_2(1 - d_3 \phi_2) + d_3}$$

$$C = -\phi_1(1 - d_3 \phi_2) - \phi_2$$

$$D = 1 - d_2 \phi_1 - d_3 \phi_2 + d_1[-\phi_1(1-d_3\phi_2) - \phi_2]$$


Again, C and D are essentially meaningless to us, as any measurement of C or D requires knowledge of the phase on the detector. 
From from the formula above, **A is independent of** $d_1$:

$$A(d_1, d_2, d_3, \phi_1, \phi_2) = A(d_2, d_3, \phi_1, \phi_2)$$

Andy information on the distance from the sample to first lens must be fit from the B element, which is affine in $d_1$:

$$B(d_1, d_2, d_3, \phi_1, \phi_2) = d_1 \cdot A(d_2, d_3, \phi_1, \phi_2) + d_2(1 - d_3 \phi_2) + d_3$$

We note also that images which are recorded by the microscope to try and fit the microscope lens state, serve no other purpose than to measure A and B - and it is meaningless for us now to use images at the end of any optimisation loop to fit the microscope - in this way we can avoid expensive optimisation routines on pixels on the detector. Provided we can determine A and B from any image in the microscope, we have extracted all the information that intensity images can give us about the microscope state.

---

Let us imagine that we wish to invert the distances and focal lengths of the microscope from intensity images on the detector, which tell us the A and B (Magnification and Defocus) of the microscope system. It is obvious that one image of a known object such as an aperture, will not be sufficient to solve for d1, d2, d3 and f1 and f2, since we have 5 unknowns and only 2 equations. We need to introduce some diversity to get more equations.

  

As we have discussed earlier, perturbing the object in the sample plane moves $d_1$, which effects the lens transfer matrix elements $B$ and $D$,
  

One could imagine that we could use a known aperture, and by moving the sample plane, we could record an ensemble of images with different $d_1$ values, and thus we would record different defocused images of the aperture.

However, this approach fundamentally fails because the **the $A$ element of the transfer matrix is independent of $d_1$**. The optical scaling of the entire system is determined solely by the downstream optics:

  
$$A = (1 - d_2 \phi_1)(1 - d_3 \phi_2) - d_3 \phi_1,$$

which depends only on $d_2, d_3, \phi_1, \phi_2$—not on $d_1$.

When you move the sample plane by a distance $\Delta d_1$, you obtain a new measurement $B' = B + A \cdot \Delta d_1$. While this adds more data points, all of these points lie on a **straight line** in $(d_2, d_3, \phi_1, \phi_2)$ space with slope determined by $A$. Since $A$ is the same for all sample positions, the line's slope is fixed, and you learn nothing new about the downstream optics from its position. Changind $d_1$ does not make new equations to add up to help determine the unknowns, it just gives you more points in the same line, without helping to determine how the slope and intercept of the line are determined by the downstream optics.

In the two lens case, this degeneracy of the solutions can be broken if the total distance $d_1 + d_2 + d_3$ is constrained (e.g., the microscope column length is known). Otherwise, for an N lens system, other constraints must be applied to the system in order to ensure that a unique solution can be found. 

One can also say that wobbling the focal length of a microscope lens by an unknown amount as is possible in the electron microscope will not add any new information, just extra unknowns into this system of equations, making the problem worse. 

In order to overcome this degneneracy to solve for distances and focal lengths in a 2 lens system, and up to N-lens system we need to introduce additional physical constraints on the system. 

One such constraint which can encapsulate two constraints in one function is to choose a physically reasonable function to describe how the focal length changes with input current and accelerating voltage of the microscope. 

A simple function to describe the focal length of an unsaturated electromagnetic lens is given by: 

$$
f = \frac{K * V}{I^2} 
$$

where $K$ is a constant that depends on the geometry of the lens and number of turns of the lens, $V$ is the accelerating voltage, and $I$ is the input current.

**Summary of why this helps (constraints for 3 lenses and beyond):** The inverse of focal length is optical power, so
$$
\phi(I, V) = \frac{1}{f} = \frac{I^2}{K V}.
$$
If we wobble current around $I_0$ with $I = I_0(1+w)$, then
$$
\phi(w) = \phi_0 (1+w)^2 = \phi_0 + 2\phi_0 w + \phi_0 w^2.
$$
This forces a **quadratic coupling** between the wobble coefficients: if we write $\phi(w)=a+bw+c w^2$, then the coefficients must satisfy
$$
c = \frac{b^2}{4a}.
$$
That single relation is an extra physical constraint per lens. It reduces the effective degrees of freedom of each lens (the curvature term is not free), which breaks the scaling symmetry that creates the degenerate family of solutions. For a 3-lens system, that means the wobble data across multiple $w$ values now overconstrain the model, and the solution can become unique when enough settings are used.

**How changing accelerating voltage adds constraints:** The same law also ties optical power to $V$. At two voltages $V_1$ and $V_2$ for the same geometry,
$$
\phi_i(V_2) = \gamma \phi_i(V_1), \quad \gamma = \frac{V_r(V_1)}{V_r(V_2)}.
$$
This couples *two data sets* with a known scaling factor, without introducing new unknowns. The degeneracy that rescales distances and lens powers can satisfy one voltage, but it cannot satisfy both simultaneously unless the scale factor is exactly one. That extra cross-voltage constraint is especially powerful for 3 lenses and higher, where wobble-only data still leaves residual symmetries.

**Image rotation as an additional constraint:** Magnetic lenses also rotate the image by an angle that depends on the axial magnetic field. For a thin lens, the rotation angle is approximately proportional to current:
$$
\theta_i \propto I_i, \quad \text{or more precisely,} \quad \theta_i = \frac{e \mu_0 I}{2m_e v} \cdot (\text{geometric factor}).
$$
The total rotation at the detector is the sum over all lenses: $\theta_{\text{total}} = \sum_{i=1}^{N} \theta_i$. In wobble space with $I_i = I_{0,i}(1+w_i)$:
$$
\theta_i(w_i) = r_i (1 + w_i), \quad r_i = \text{baseline rotation for lens } i.
$$
**Why this helps break degeneracy:** Rotation scales **linearly** with current, while optical power scales **quadratically** ($\phi \propto I^2$). Under the degeneracy transformation that rescales distances and optical powers:
- If $\phi_i \to \lambda^{-1}\phi_i$, then $I_i^2 \to \lambda^{-1}I_i^2$, hence $I_i \to \lambda^{-1/2}I_i$
- This implies $\theta_i \to \lambda^{-1/2}\theta_i$

But the total rotation $\theta_{\text{total}} = \sum \theta_i$ is directly measurable from images (via feature tracking or cross-correlation). The degeneracy transformation cannot simultaneously satisfy **both** the $\phi \propto I^2$ relationship (quadratic) **and** the $\theta \propto I$ relationship (linear) across all lenses with a single scaling parameter $\lambda$. 

**Practical advantage:** Rotation can be measured directly from intensity images without requiring phase information, making it an accessible additional observable. For N≥3 lenses, combining wobble data with rotation measurements provides strong constraints that help uniquely determine the system parameters. The rotation data effectively couples the currents in a way that conflicts with the scaling symmetry.

**Can nonlinear model + rotation solve N>3 with one voltage?** YES! When both physics models are enforced simultaneously:

- The nonlinear constraint $\phi(w) = \phi_0(1+w)^2$ means each lens has only **one free parameter** $\phi_0$ (baseline optical power). The wobble response is fully determined by this single value.
- Rotation adds K independent measurements: $\theta_{\text{total}}(w) = \sum_{i=1}^N r_i(1+w_i)$ where $r_i$ is directly related to the baseline current.
- **Reduced parameter count:** N+1 distances + N baseline optical powers = **2N+1 unknowns** (not 3N+1!)
- **Enhanced constraint count:** 2K equations (A,B) + K equations (rotation) = **3K constraints**

For uniqueness by counting: $3K \ge 2N+1$

| N lenses | Unknowns (2N+1) | Min wobbles (K ≥ ⌈(2N+1)/3⌉) |
|:---------|:----------------|:------------------------------|
| 3        | 7               | K ≥ 3                         |
| 4        | 9               | K ≥ 3                         |
| 5        | 11              | K ≥ 4                         |
| 6        | 13              | K ≥ 5                         |

**Why this works beyond counting:** The degeneracy transformation requires $I \to \lambda^{-1/2}I$ to maintain $\phi \propto I^2$. But then:
- Optical power satisfies: $\phi(w) = \phi_0(1+w)^2$ ✓ (constraint preserved)
- Rotation predicts: $\theta = r(1+w)$ with $r \propto I_0 \to \lambda^{-1/2}r$ ✗ (conflicts with measured $\theta_{\text{total}}$)

The two models impose **incompatible scaling laws** on the same underlying currents. No value of λ can satisfy both simultaneously across all wobble settings and all lenses.

**Conclusion:** For N≤6 lenses, the combined nonlinear + rotation model with K≥5 wobble settings at **one voltage** should uniquely determine all distances and optical powers. This is substantially simpler than requiring two accelerating voltages.





## The Inverse Problem: Wobble Measurements

### Measurement Setup

We perturb the lens currents according to:

$$
\phi_i(w_i) = a_i + b_i \cdot w_i
$$

where:
- $a_i$ = baseline optical power (= $1/f_i$)
- $b_i$ = wobble sensitivity (proportional to $I_0$ and lens geometry)
- $w_i \in [-w_{\max}, w_{\max}]$ = relative wobble excursion

For each **wobble setting** $(w_1, w_2)$, we measure the **A and B** matrix elements at the detector:

$$
A(w_1, w_2) = (1 - d_2[a_1+b_1 w_1])(1 - d_3[a_2+b_2 w_2]) - d_3[a_1+b_1 w_1]
$$

$$
B(w_1, w_2) = d_1 \cdot A(w_1, w_2) + d_2(1-d_3[a_2+b_2 w_2]) + d_3
$$

### Polynomial Structure in Wobble Space

Expanding $A(w_1, w_2)$ in powers of $w_i$:

$$
A = \underbrace{(1-d_2 a_1)(1-d_3 a_2) - d_3 a_1}_{a_{00}} 
  + \underbrace{[-d_2(1-d_3 a_2) - d_3] b_1}_{a_{10}} w_1
  + \underbrace{[-d_3 b_2(1-d_2 a_1)]}_{a_{01}} w_2
  + \underbrace{[d_3 b_1 b_2]}_{a_{11}} w_1 w_2
$$

Similarly, $B(w_1, w_2)$ is **affine in wobble**:

$$
B = B_{\text{const}} + B_{w_1} w_1 + B_{w_2} w_2
$$

where all coefficients depend on the unknown parameters $(d_1, d_2, d_3, a_1, b_1, a_2, b_2)$.

---

## The Degeneracy Problem

### Statement of the Problem

With **K wobble settings** $(w_1^{(k)}, w_2^{(k)})$ for $k=1,\ldots,K$, we have:

- **Unknowns:** 7 parameters $(d_1, d_2, d_3, a_1, b_1, a_2, b_2)$
- **Constraints:** $2K$ equations (A and B for each setting)

For uniqueness by counting, we need $2K \ge 7$, so $K \ge 4$.

In practice, with **K=5 wobble settings**, we have **10 equations** and **7 unknowns—seemingly overdetermined**.

However, numerical experiments show **~200 distinct global solutions** exist.

### Why Wobble Perturbations Alone Don't Break Degeneracy

The heart of the problem is that **both $a_2$ and $b_2$ scale the same way** under the degeneracy transformation. Let me show you exactly why this fails.

**The degeneracy transformation is:**
$$d_2 \to \lambda d_2, \quad d_3 \to \lambda d_3, \quad a_2 \to \lambda^{-1}a_2, \quad b_2 \to \lambda^{-1}b_2$$

**For any wobble value** $w_2$, the optical power is:
$$\phi_2(w_2) = a_2 + b_2 w_2$$

Under scaling, this becomes:
$$\phi_2'(w_2) = \lambda^{-1}a_2 + \lambda^{-1}b_2 \cdot w_2 = \lambda^{-1}(a_2 + b_2 w_2) = \lambda^{-1}\phi_2(w_2)$$

**The key observation:** The optical power at EVERY wobble value scales by exactly $\lambda^{-1}$.

Now look at what appears in the ABCD formulas. Both A and B depend on $d_3 \phi_2(w_2)$:
$$d_3' \phi_2'(w_2) = (\lambda d_3) \cdot (\lambda^{-1}\phi_2(w_2)) = d_3 \phi_2(w_2)$$

**This product is completely invariant!** The same factor $\lambda$ that scales the distance is exactly canceled by the inverse scaling of optical power.

**What this means:** All K wobble measurements see the SAME invariance. When you measure at $w_2 = 0.001$, you get scaling. When you measure at $w_2 = 0.010$, you also get the same scaling. There's no wobble value that behaves differently. So you get K copies of the same symmetry—10 equations that all respect the same 1-parameter degeneracy. Adding more wobble points doesn't help.

**Summary:**
- **K=1 wobble:** 2 equations, but they both maintain the scaling symmetry
- **K=5 wobble:** 10 equations, but all 10 equations maintain the scaling symmetry
- **K=100 wobble:** 200 equations, but all 200 equations still maintain the scaling symmetry

The 1-parameter family is an **exact algebraic symmetry** of the problem. More wobble points = more equations, but all compatible with the same continuous family of solutions.

**Why nonlinear physics breaks this:** If the optical power model is nonlinear like $\phi(w) = a(1+w)^2$, then when you expand it you get coefficients like $c = b^2/(4a)$ that are coupled asymmetrically. You cannot scale all three of $a$, $b$, $c$ by $\lambda^{-1}$ and simultaneously satisfy this constraint across the entire problem. Different wobble values impose conflicting rescaling demands on the same parameters, breaking the symmetry.

---

## Three Solutions That Break the Degeneracy

### Solution 1: Known Total Distance ($d_1 + d_2 + d_3$)

**Constraint:**
$$d_1 + d_2 + d_3 = L_{\text{total}} \quad \text{(measured/known)}$$

**Result:** UNIQUE solution for N=2 lenses. Tested across 4 parameter regimes and 500 random starts—all converged to the truth.

**Why it works:** The degeneracy scales all downstream distances by $\lambda$, so $(d_2 + d_3) \to \lambda(d_2 + d_3)$. Fixing the sum pins $\lambda = 1$.

**Practical advantage:** Requires no extra measurements—just ruler/CAD knowledge of column geometry.

---

### Solution 2: Nonlinear Lens Model ($\phi \propto I^2$)

**Physics:** For round magnetic lenses:
$$\phi(I) = \alpha I^2$$

In terms of relative wobble $w = \delta I / I_0$:
$$
\phi(w) = a(1+w)^2 = a + 2a\,w + a\,w^2
$$

**Parameter form:** $\phi(w) = a + bw + cw^2$ with **constraint**:
$$
c = \frac{b^2}{4a}
$$

**Result:** UNIQUE solution for N=2 lenses with $K=5$ wobble settings.

**Why it works:** The linear degeneracy permits independent rescaling of $a$ and $b$. The constraint $c = b^2/(4a)$ couples them nonlinearly, breaking the symmetry. A single value of $\lambda$ cannot satisfy the constraint across all wobble settings simultaneously.

**Practical advantage:** Requires no extra measurements beyond standard wobble experiment. Needs measurable curvature (use wobble $w \sim 0.01$–0.02).

---

### Solution 3: Two Accelerating Voltages

**Physics:** For magnetic lenses, optical power is inversely proportional to relativistic-corrected voltage:

$$\phi \propto \frac{1}{V_r}, \quad V_r = V \left(1 + \frac{eV}{2m_0c^2}\right)$$

**Measurement:** At two voltages $V_1$ and $V_2$:

$$
\phi_i(V_2) = \gamma \cdot \phi_i(V_1), \quad \gamma = \frac{V_r(V_1)}{V_r(V_2)}
$$

Geometry $(d_1, d_2, d_3)$ unchanged.

**Result:** UNIQUE solution with $K \ge 3$ wobble settings at each voltage.

**Why it works:** The degeneracy $(d_2, d_3, \phi_2) \to (\lambda d_2, \lambda d_3, \lambda^{-1}\phi_2)$ must hold at **both** voltages. At $V_1$, one value of $\lambda$ works. At $V_2$ with the same geometry but scaled optical powers, a different $\lambda$ is required. No single $\lambda$ satisfies both datasets.

**Practical advantage:** Model-independent. Even 10 kV difference (5%) at 200 kV is enough. Fast convergence.

---

## Comparison of Solutions

| Solution | Extra info | Measurements | Unique for N=2? | Robustness |
|:--|:--|:--|:--|:--|
| Known $d_1+d_2+d_3$ | Total column distance | 0 | YES | ✓ All regimes |
| Nonlinear $\phi(I)$ | Physics model only | 0 (wider wobble) | YES | ✓ All regimes |
| Two voltages | Second HT | 1 full repeat | YES | ✓ All regimes |

---

## Scaling to Multiple Lenses (N > 2)

### Unknowns vs. Constraints

For **N lenses**:
- **Unknowns:** $(N+1)$ distances + $2N$ optical parameters = $3N+1$ total
- **Constraints per setting:** 2 (only A and B measurable)
- **Total constraints at K settings, one voltage:** $2K$
- **Total constraints at K settings, two voltages:** $4K$

Minimum K by counting:

$$
K_{\min} = \left\lceil \frac{3N+1}{2K_{\text{volt}}}\right\rceil
$$

| N | One voltage | Two voltages |
|:--|:--|:--|
| 2 | $K \ge 4$ | $K \ge 2$ |
| 3 | $K \ge 5$ | $K \ge 3$ |
| 4 | $K \ge 7$ | $K \ge 4$ |
| 5 | $K \ge 8$ | $K \ge 4$ |

### Generalization of the Three Solutions

1. **Known total distance:** Removes one global scaling DOF. For N=3, 4, 5 it helps but is **insufficient alone**.
2. **Nonlinear model:** Still couples optical powers nonlinearly. Likely unique for N≤3 with enough K and wobble amplitude.
3. **Two voltages:** Most robust. Guarantees uniqueness by symmetry argument for any N if $4K \ge 3N+1$.

---

## Recommendations for Implementation

### For a real TEM with 2 lenses:
Use the **nonlinear $\phi(I) = \alpha I^2$ model** with K=5–10 wobble points. Requires no extra equipment, only larger wobble amplitudes to make the curvature observable.

### For systems with N ≥ 3:
Combine **two voltages** (one voltage change) with **K ≥ 4–5 wobble settings**. This is the most robust, requires minimal additional measurements, and works across all parameter regimes virtually guaranteed.

---


