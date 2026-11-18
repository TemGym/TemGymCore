"""Read every lens and deflector value at every selector position of the console. No beam needed.

Steps through MAG, LowMAG, SAMAG and DIFF (every row), and every spot size and alpha at the row the console is on.
Saves console_tables.json next to this script. 

    python record_console_tables.py
"""
import json
import time

from PyJEM import TEM3

lens, deflector, eos, ht, stage = TEM3.Lens3(), TEM3.Def3(), TEM3.EOS3(), TEM3.HT3(), TEM3.Stage3()

LENSES = ["CL1", "CL3", "CM", "OLc", "OLf", "OM", "IL1", "IL2", "IL3", "IL4", "PL1"]
DEFLECTORS = ["GunA1", "GunA2", "SpotA", "CLA1", "CLA2", "CLs", "ShifBal", "TiltBal", "AngBal",
              "OLs", "ILs", "IS1", "IS2", "PLA"]
MODES = {0: "MAG", 2: "LowMAG", 3: "SAMAG", 4: "DIFF"}
WAIT_S = 0.1            # after changing a selector


def read():
    """Everything the console has now."""
    values = dict(mode=eos.GetFunctionMode(), mag=eos.GetMagValue(), selector=eos.GetCurrentMagSelectorID(),
                  spot=eos.GetSpotSize(), alpha=eos.GetAlpha(), ht=ht.GetHtValue(), stage=stage.GetPos())
    for name in LENSES:
        values[name] = getattr(lens, "Get" + name)()
    for name in DEFLECTORS:
        values[name] = getattr(deflector, "Get" + name)()
    return values


start = read()
print("start:", start["mode"], start["mag"], "spot", start["spot"], "alpha", start["alpha"])
deflector.SetBeamBlank(1)
tables = dict(start=start, rows={}, spot_alpha=[])

# Every selector row in each function mode
for number, name in MODES.items():
    eos.SelectFunctionMode(number)
    time.sleep(WAIT_S)
    tables["rows"][name] = []
    index = 0
    while True:
        try:
            eos.SetSelector(index)
        except Exception:
            break                                   # past the last row
        time.sleep(WAIT_S)
        values = read()
        if tables["rows"][name] and values["selector"] == tables["rows"][name][-1]["selector"]:
            break                                   # the selector stopped moving: the last row
        tables["rows"][name].append(values)
        print(name, values["mag"], {n: values[n] for n in ("IL1", "IL2", "IL3", "PL1")})
        index += 1

# Every spot at every alpha, at the mode and row the console started on
eos.SelectFunctionMode(start["mode"][0])
eos.SetSelector(start["selector"][0])
time.sleep(WAIT_S)
for alpha in range(9):
    try:
        eos.SetAlphaSelector(alpha)
    except Exception:
        continue                                    # this alpha does not exist
    for spot in range(8):
        eos.SelectSpotSize(spot)
        time.sleep(WAIT_S)
        values = read()
        tables["spot_alpha"].append(values)
        print("alpha", alpha, "spot", spot, {n: values[n] for n in ("CL1", "CL3", "CM")})

# Back to where the console was
eos.SetAlphaSelector(start["alpha"])
eos.SelectSpotSize(start["spot"])
eos.SelectFunctionMode(start["mode"][0])
eos.SetSelector(start["selector"][0])
deflector.SetBeamBlank(0)

with open("console_tables.json", "w") as file:
    json.dump(tables, file, indent=1, default=str)
print("saved console_tables.json")
