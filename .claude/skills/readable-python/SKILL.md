---
name: readable-python
description: How Python code, comments, docstrings and notebooks are written in this repository so a scientist can read them top to bottom. Use it before writing or editing any .py file or notebook here, and when reviewing one.
---

# Readable Python

The reader is a physicist who knows Python, NumPy and electron optics, has not seen the
conversation, and reads the file from the top. Code is done when that reader can follow it
without asking what a name, a number or a step is for.

The sources, in order when they disagree: this file, `CONTRIBUTING.md`, PEP 8, PEP 257,
the numpydoc docstring guide, the Google Python Style Guide (its sections on comments and
naming). `ruff` settles formatting; do not argue with it.

## Code

1. **Straight-line first.** A script or notebook cell reads as the steps of the calculation,
   in order. No class, wrapper, registry, config object or helper layer until the same code
   has been needed in three places.
2. **A function earns its place** by being called more than once or by naming a real step of
   the physics. A function called once, whose body is shorter than its docstring, is inlined.
3. **Names say the physical quantity**, with the unit where it is not SI or not obvious:
   `camera_length_m`, `kick_px_per_dac`, `cl3_dac`. No `tmp`, `data2`, `res`, `helper`,
   `process`, `handle`. Loop letters `i`, `j`, `k` and the field's symbols (`f`, `z`, `theta`)
   are fine.
4. **Every literal number has a name or a comment with its source**: measured (where, when),
   from a document (which), or chosen (why this and not ten times more).
5. **One way to do each thing.** Use what the file and the library already use; look before
   writing a mechanism. No dependency is added without asking.
6. **No defensive code for cases nobody has hit**: no try/except around code that cannot
   fail, no `isinstance` ladders, no optional arguments nobody passes.
7. **Line length 100**, imports at the top in three groups (standard library, third party,
   this package), no unused imports or variables, no commented-out code.
8. **Delete rather than keep.** A superseded function or cell is removed in the same change;
   git has the old one.

## Comments

A comment says what the code cannot: **why** it is this way, **where a number came from**,
**what would go wrong** if it were changed, or **what the physics is**. It is written for
someone reading the file a year from now.

Never write:

- a restatement of the line (`# loop over the lenses`, `# compute the misfit`);
- the story of the session (`# changed from 32768`, `# new version`, `# fixed`, `# as discussed`,
  `# now uses`); history belongs in the commit message;
- section banners of dashes or equals signs; a blank line and one plain sentence do it;
- a comment on every line; most lines need none;
- hedges and praise (`# simple helper`, `# robust`, `# note that`, `# importantly`).

### How a comment is worded

Comments, docstrings and notebook markdown cells follow the structural rules of Simplified
Technical English (ASD-STE100, as in the `asd-ste100` skill, strict mode). The rules:

1. **Full sentences.** Each sentence has its subject, its verb and its articles, and ends
   with a full stop. No fragments, no dropped words.
2. **One idea per sentence.** At most 25 words. Two ideas are two sentences.
3. **Active voice, simple tenses.** "The fit holds CM constant.", not "CM is held constant"
   and not "CM has been held constant".
4. **No semicolons, and no colon or dash that joins two statements.** Write two sentences.
5. **No rhetorical shapes.** No "X, not Y", no "not just X but Y", no "hence", "thus",
   "note that", "simply". State the fact.
6. **One name for one thing**, the same name the code uses. If the variable is
   `field_length`, the comment does not say "width" or "extent".
7. **No two-word verbs** ("set up", "work out", "end up"). Use the one plain verb.
8. **A verb, not a noun made from a verb.** "The fit adjusts CL3.", not "The fit performs
   an adjustment of CL3."
9. **No more than three nouns in a row.**
10. **Keep the doubt the facts have.** "may", "about" and "is not known" stay. A comment
    does not state as fact what was only assumed.

Write:

```python
# Each lens has its own middle DAC value. With one value for all lenses, the fit cannot
# separate the middle value from phi_a.
MIDDLE_DAC = (32768.0, 32768.0, 32768.0, 31593.0, 43363.0, 31047.0, 61029.5)

# No measurement gives the field length of CM. Lengths from 2 mm to 10 mm fit the measured
# focus curve equally well. This model uses 5 mm.
CM_FIELD_LENGTH_M = 5e-3

# The image rotation increases in proportion to the lens current. The lens power increases
# in proportion to the square of the lens current (Busch).
power = theta**2 * phi / w
```

Not:

```python
# Set the middle DAC values
# 5 mm is a choice, not a measurement: anything from 2 to 10 mm fits the focus curve equally.
# Rotation is linear in excitation, focusing power quadratic (Busch), hence theta**2.
```

When code changes, read its comments again and change them with it. A comment that is no
longer true is deleted.

## Docstrings

- Public functions and classes in `src/`: numpydoc, with units for every argument and
  return value (`CONTRIBUTING.md`).
- Functions in notebooks and scripts: one sentence saying what goes in and what comes out,
  with units. No `Parameters` block for a three-line function.
- The first line is a full sentence in the imperative or a plain statement, ending in a full
  stop. It does not repeat the function's name.

## Notebooks

1. **Runs top to bottom** on a fresh kernel, with no hidden state and no cell that must be
   run twice. Run it that way before saying it is done.
2. **One idea per cell.** Before each code cell or group, a markdown cell of one to three
   plain sentences: what this step does and what to look for in its output.
3. **Order:** the numbers, the model, one worked example, then the checks. Imports and
   constants in the first code cell.
4. **Outputs say what they are**, with units: `print(f"kept-back rows: {rms_px:.2f} px rms")`,
   axis labels on every figure.
5. **No cell longer than a screen** (about 40 lines). A longer one is two steps.
6. **Slow cells** say how long they take and are behind a flag that loads the saved result.
7. No dead cells, no scratch experiments, no cell that only redefines an earlier one.

## Before handing over

- Reread the diff as the reader described at the top.
- Run `ruff check` on changed `.py` files and the notebook top to bottom; report the
  commands and their output.
- Run `python .claude/skills/readable-python/scripts/check_comments.py <changed files>`
  (`.py` and `.ipynb`). Correct every finding of the level "fix". Read every finding of the
  level "look" and correct it unless the sentence is clearer as it is. Report the last line
  of the output.
- List every literal number added, with its source.
