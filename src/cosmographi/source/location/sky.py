from ..base import Source
import jax.numpy as jnp

class Sky:
    """A class to hold all of the located objects

    Parameters
    ----------
    Objects: jnp.ndarray[tuple[float], list[Source]]
        All objects in the sky. The dictionary is of the form (right assension, declination angle): list[Source]
    """

    def __init__(self, objs: dict|  None = None) -> None:
        """Initialize this sky. Objs should be in the same format as the objects parameter defined for this class."""
        if objs is None:
            self.objects = jnp.ndarray()
        else:
            self.objects = objs
    
    def add_object(self, ra: float, dec: float, source: Source) -> None:
        coor = (ra, dec)
        if coor not in self.objects:
            self.objects[coor] = jnp.asarray([source])
        else:
            self.objects[coor].append(source)
    
    def get_objects_in_region(self, start_ra: float, end_ra: float, start_dec: float, end_dec: float) -> jnp.ndarray[Source]: 
        """Assume pixel covers a square region from start to end ra and dec. Return all sources that are w"""


        
