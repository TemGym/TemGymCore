"""Stage 2 (drafts, on hold): the physical projection system.

Two specimen frames (a cross grating at the 40k row for |A|, gold rings at the DIFF row for |B|)
turn deflector units into metres and radians, and the response at a second HT fixes the lens
and camera positions. Fitted together with the stage-1 recording.

* :mod:`stage2.record` -- at the microscope (PyJEM): the specimen frames and the second-HT recording.
* :mod:`stage2.specimen_frames` -- |A| from the grating image, |B| from the gold rings.
* :mod:`stage2.joint_fit` -- everything fitted together; lens and camera positions with uncertainties.
* :mod:`stage2.figures` -- the column drawn with ring lenses; the specimen frames.
* :mod:`stage2.stage1_fit`, :mod:`stage2.projection_lenses`, :mod:`stage2.neoarm`, :mod:`stage2.stage1_figures` --
  stage 2's own copies of the stage-1 pieces (the stage-1 notebook has the versions in use).
"""
