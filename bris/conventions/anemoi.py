from . import cf

def get_units(name):
    """Returns the units used in Anemoi Datasets"""

    # Assume anemoi datasets use CF units
    cfname = cf.get_metadata(name)["cfname"]
    units = cf.get_attributes(cfname)["units"]

    # Here's an opportunity to override, if needed:
    # if name == "2t":
    #   return "degC"

    return units
