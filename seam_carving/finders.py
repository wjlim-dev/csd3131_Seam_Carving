"""Name -> seam finder lookup, used by the CLI, benchmarks and the app."""

from importlib import import_module

# Modules are imported only when asked for, so an unfinished finder
# never breaks the others.
FINDERS = {
    "dummy": "seam_carving.dummy",
    "dp": "seam_carving.dp",
    "greedy": "seam_carving.greedy",
    "graphcut": "seam_carving.graphcut",
}


def get_finder(name):
    if name not in FINDERS:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from: {', '.join(FINDERS)}")
    return import_module(FINDERS[name]).find_vertical_seam
