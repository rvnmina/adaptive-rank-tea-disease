import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "ARB_ROOT",
    os.path.dirname(HERE) if os.path.basename(HERE) == "code" else HERE
)
RUNS  = os.environ.get("ARB_OUT",   os.path.join(ROOT, "runs"))
RES   = os.environ.get("ARB_RES",   os.path.join(ROOT, "results"))
FIG   = os.environ.get("ARB_FIG",   os.path.join(ROOT, "figures"))
CACHE = os.environ.get("ARB_CACHE", os.path.join(ROOT, "cache"))
PV_ROOT = "/home/ai25ai60r02/tea_dataset/Tea Leaf Disease Image"