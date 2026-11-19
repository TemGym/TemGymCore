# Working in TemGymCore

DRAFT for David to edit. Rules here load in every session.

## Code

- Before writing or editing any `.py` file or notebook, follow the `readable-python` skill
  (`.claude/skills/readable-python/SKILL.md`).
- Before writing or editing any code that models optics, follow the `temgym` skill
  (`.claude/skills/temgym/SKILL.md`): TemGym components and tracing functions only, and stop
  and ask when TemGym lacks what is needed.
- Before writing or editing any code that reads or changes the microscope, follow the `pyjem`
  skill (`.claude/skills/pyjem/SKILL.md`): jemwrap's control functions, no microscope object,
  rehearsed on the simulated microscope first.
- Numerical code is JAX. No scipy in new code under `examples/`.
- No wrapper layers: a notebook is plain TemGym and JAX calls a reader can follow.

## Process

- A larger change starts with `/kaizen:playbook`; requirements and acceptance examples live
  in `design/`.
- A change of more than a few hundred lines is reviewed by an agent that did not write it
  before it is handed over.
- Commit only when asked.

## Do not touch

- `examples/microscope_inversion/neoarm_forward.py`, `neoarm_forward.ipynb`,
  `neoarm_predictor.ipynb` (another agent's files).
