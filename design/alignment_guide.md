  
**NeoARM TEM-L Parallel Beam Alignment Guide**

These alignments are followed sequentially, but the Gun and Condenser Pivot Point sections are each iterative. The whole set of alignments should likely be looped over more than once, in case the starting point was too far from the final condition. The general flow of alignments is top-to-bottom in the column, with the exception of aperture centering and HT Wobbler for Beam Tilt (CLA2).

---

**Load existing alignments, check Imaging Mode**

In JEOL’s TEM Center, at the bottom of **Alignment Panel**, subsection *Backup*, use the button “Load” to load an existing, recent alignment file for TEM.

Once loaded, check in the left column, that the **Mode** is TEM, and the **Alpha** is TEM-L.

---

These initial TEM Alignments use the **100 μm condenser aperture**

**Aperture centering**  
Unclear *when exactly* to centre the aperture, likely must be iterated / checked after each alignment below.

* GOAL: Centre the aperture  
* CONDITION:  
  * Centre beam with BEAM SHIFT (CLA1) to aid the centre  
  * 50K Mag  
* METHOD:  
  * Change CL3 / Brightness to converge and expand the beam  
  * Expand the beam beam each side of crossover, measure the centre point  
  * Adjust the aperture position towards the average of the two beam centres until convergence  
* ENDPOINT: passing through crossover doesn’t shift the beam

---

**Gun Alignments**  
Iterate between tilt and shift

**1 \- Gun Tilt**

* GOAL: Beam enters CL1 after the gun parallel to the optical axis  
* CONDITION:  
  * Spot size 1  
  * 50K Mag  
  * Centre the beam with BEAM SHIFT (CLA1)  
* METHOD:  
  * Wobble Anode ⇒ beam will oscillate in size and shift around the screen  
  * Adjust Gun Tilt X/Y (GUN2 X/Y)  
* ENDPOINT: beam oscillates in size around a fixed point

**2 \- Gun Shift**

* GOAL: Beam enters CL1 after the gun on the optical axis  
* CONDITION:  
  * Spot size 5  
  * 25K initial Mag  
  * Converge beam to a point with Brightness / CL3  
* METHOD-ITERATE:  
  * Spot size 5  
  * Centre the beam with BEAM SHIFT (CLA1)  
    ————  
  * Spot size 1  
  * Centre the beam with GUN SHIFT (GUN1)  
    ————  
  * Increase Mag progressively until 200K  
* ENDPOINT: beam does not move much when changing spot size 1 ⇒ 5  
* OUTER-ITERATE: Repeat / Check **Gun Tilt Alignment**

---

**Spot Alignment**  
No procedure for this???  
Only one deflector X/Y setting for this  
Does not change substantially between spot sizes, factory setting? How to check it is correct.  
NOTE: when changing spot size, CL1, CL3, SPOT ALIGN and COND STIG all change. CL3 value is some fixed value for each spot size plus a “hidden” offset for the current CL3 brightness setting.

---

**Corrector Beam Tilt**  
If at this stage the beam intensity is heterogeneous (concentrated towards one direction while condensed), then it can be useful to adjust the beam tilt of the corrector.

* GOAL: Beam has a homogeneous angular intensity distribution (not necessarily radially)  
* CONDITION:  
  * Imaging mode  
  * 50K Mag, progressive  
* METHOD:  
  * Within ASCOR (corrector software) adjust **User Beam Tilt X/Y**  
  * Changing the Beam Tile setting will shift the beam on the screen ⇒ bring it back to the centre using **BEAM SHIFT (CLA1)**  
* ENDPOINT: beam angular intensity distribution uniform up to a high magnification / condenser setting

---

**Condenser Astigmatism**

* GOAL: Beam is circular  
* CONDITION:  
  * 200K Mag  
* METHOD:  
  * Converge the beam with Brightness / CL3  
  * Adjust Condenser Stig X/Y to make the spot circular  
  * Check both sides of crossover (converge through BRIGHTNESS / CL3)

---

**Condenser Pivot Points**  
Iterate between tilt and shift

**1 \- Pivot Points: Tilt Balance:**

* GOAL: The beam can Tilt on the sample plane without Shifting  
* CONDITION:  
  * Imaging Mode  
  * Centre beam with BEAM SHIFT / CLA1  
  * Converged beam with Brightness / CL3  
  * Initial magnification appropriate to wobbling strength and current alignment (200K)  
* METHOD \- ITERATE:  
  * Activate Tilt X wobbler ⇒ the beam oscillates in X  
  * Adjust TILT BALANCE X to minimise the oscillation  
  * Deactivate Tilt X wobbler  
    ————  
  * Activate Tilt Y wobbler ⇒ the beam oscillates in Y  
  * Adjust TILT BALANCE Y to minimise the oscillation  
  * Deactivate Tilt Y wobbler  
    ————  
  * Increase Magnification and repeat  
* ENDPOINT: The wobbling beam is stationary in X / Y to a high magnification

**2 \- Pivot Point: Shift Balance:**

* GOAL: The beam can shift on the sample plane without tilting.  
* CONDITION:  
  * Diffraction Mode  
  * Minimise diffraction spot with Brightness / CL3 (anti-clockwise)  
    * NOTE: Instructions say “Turn completely anti-clockwise CL3”, then use DIFF FOCUS. To-check ⇒ if really fully anti-clockwise?  
  * Converged diffraction spot caustic with DIFF FOCUS (IL1)  
* METHOD-ITERATE:   
  * Activate Shift X wobbler ⇒ the beam oscillates in X  
  * Adjust SHIFT BALANCE X to minimise the oscillation  
  * Deactivate Shift X wobbler  
    ————  
  * Activate Shift Y wobbler ⇒ the beam oscillates in Y  
  * Adjust SHIFT BALANCE Y to minimise the oscillation  
  * Deactivate Shift Y wobbler  
    ————  
  * Increase Camera Length and repeat  
* NOTE:  
  * If the oscillation is oval in shape, adjust ANGLE BALANCE X / Y to squash the oval into a line, then minimise with SHIFT BALANCE X / Y  
* ENDPOINT: The wobbling beam is stationary in X / Y to a high camera length  
* OUTER-ITERATE: Repeat / Check **Condenser Pivot Points Tilt**

---

**Condenser Alignment Tilt (CLA2) \- HT Wobbler**

* GOAL: Beam is parallel to optical axis when entering OL  
* CONDITION:  
  * 500K Mag  
  * Characteristic feature of sample is imaged and at correct Z height  
  * STD FOCUS (Neutral OL-Fine)  
* METHOD:  
  * HT Wobbler ⇒ Beam and Features in image oscillate in position and size  
  * Adjust BEAM-TILT X / Y (CLA2) until Features in image oscillate in size but not position  
  * NOTE: Beam itself may continue to move even when correctly aligned, but the features should stay stationary

---

**Objective Stigmator**

* GOAL: Minimise Objective lens astigmatism  
* CONDITION:  
  * 500K+ Mag  
  * Amorphous region of the sample. What to do if no amorphous region is available?  
  * Imaging Mode  
* METHOD:  
  * Display FFT of the image  
  * Use OL-STIG X/Y to make the FFT rings circular

---

**Diff focus**  
Unclear how this interacts with the alignment for shift pivot points, how to decouple them?

* GOAL: Diffraction plane is properly imaged by IL1 in diff mode  
* ENDPOINT: Transmitted beam in the diffraction pattern is stationary under real-space beam shift  
* CONDITION:  
  * 50K Mag  
  * Diffraction Mode  
  * Point-like diffraction spots (with CL3 / Brightness)  
* METHOD:  
  * Activate Shift Wobbler X/Y together (or activate NanoMegas scanning)  
  * Use DIFF FOCUS to minimise the movement of the transmitted beam

---

**Image Focus (Sample height)**

* GOAL: Ensure that the sample plane is at the centre of the objective lens field (and therefore “in focus”.  
* CONDITION:  
  * MAG mode, appropriate magnification for features  
  * Above alignments completed / satisfactory  
  * Wide beam to cover most of screen, with features visible  
* METHOD:  
  * Adjust sample Z to minimise contrast in the image  
  * Activate Wobblers Image X / Image Y sequentially, remaining contrast should not move  
* ENDPOINT: Minimised contrast in image of sample

---

**EDS Acquisition Guide**