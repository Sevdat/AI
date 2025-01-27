from __future__ import annotations
from numpy.random._generator import Generator
import numpy.random._mt19937
from numpy.random._mt19937 import MT19937
import numpy.random._pcg64
from numpy.random._pcg64 import PCG64
from numpy.random._pcg64 import PCG64DXSM
import numpy.random._philox
from numpy.random._philox import Philox
import numpy.random._sfc64
from numpy.random._sfc64 import SFC64
from numpy.random.mtrand import RandomState
__all__ = ['BitGenerators', 'Generator', 'MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'RandomState', 'SFC64']
def __bit_generator_ctor(bit_generator_name = 'MT19937'):
    """
    
        Pickling helper function that returns a bit generator object
    
        Parameters
        ----------
        bit_generator_name : str
            String containing the name of the BitGenerator
    
        Returns
        -------
        bit_generator : BitGenerator
            BitGenerator instance
        
    """
def __generator_ctor(bit_generator_name = 'MT19937', bit_generator_ctor = __bit_generator_ctor):
    """
    
        Pickling helper function that returns a Generator object
    
        Parameters
        ----------
        bit_generator_name : str
            String containing the core BitGenerator's name
        bit_generator_ctor : callable, optional
            Callable function that takes bit_generator_name as its only argument
            and returns an instantized bit generator.
    
        Returns
        -------
        rg : Generator
            Generator using the named core BitGenerator
        
    """
def __randomstate_ctor(bit_generator_name = 'MT19937', bit_generator_ctor = __bit_generator_ctor):
    """
    
        Pickling helper function that returns a legacy RandomState-like object
    
        Parameters
        ----------
        bit_generator_name : str
            String containing the core BitGenerator's name
        bit_generator_ctor : callable, optional
            Callable function that takes bit_generator_name as its only argument
            and returns an instantized bit generator.
    
        Returns
        -------
        rs : RandomState
            Legacy RandomState using the named core BitGenerator
        
    """
BitGenerators: dict = {'MT19937': numpy.random._mt19937.MT19937, 'PCG64': numpy.random._pcg64.PCG64, 'PCG64DXSM': numpy.random._pcg64.PCG64DXSM, 'Philox': numpy.random._philox.Philox, 'SFC64': numpy.random._sfc64.SFC64}
