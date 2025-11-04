from . import cf


def get_units(name: str) -> str | None:
    """Returns the units used in Anemoi Datasets"""

    # Assume anemoi datasets use CF units
    cfname = cf.get_metadata(name)["cfname"]
    attrs = cf.get_attributes(cfname)
    units = attrs.get("units", None)

    # Here's an opportunity to override, if needed:
    if name in ["tp", "tp_acc"]:
        return "Mg/m^2"

    return units
