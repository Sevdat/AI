from __future__ import annotations
import enum
from enum import Enum
import typing
__all__: list = ['show']
class DisplayModes(enum.Enum):
    """
    An enumeration.
    """
    dicts: typing.ClassVar[DisplayModes]  # value = <DisplayModes.dicts: 'dicts'>
    stdout: typing.ClassVar[DisplayModes]  # value = <DisplayModes.stdout: 'stdout'>
def _check_pyyaml():
    ...
def _cleanup(d):
    """
    
        Removes empty values in a `dict` recursively
        This ensures we remove values that Meson could not provide to CONFIG
        
    """
def show(mode = 'stdout'):
    """
    
        Show libraries and system information on which NumPy was built
        and is being used
    
        Parameters
        ----------
        mode : {`'stdout'`, `'dicts'`}, optional.
            Indicates how to display the config information.
            `'stdout'` prints to console, `'dicts'` returns a dictionary
            of the configuration.
    
        Returns
        -------
        out : {`dict`, `None`}
            If mode is `'dicts'`, a dict is returned, else None
    
        See Also
        --------
        get_include : Returns the directory containing NumPy C
                      header files.
    
        Notes
        -----
        1. The `'stdout'` mode will give more readable
           output if ``pyyaml`` is installed
    
        
    """
CONFIG: dict = {'Compilers': {'c': {'name': 'msvc', 'linker': 'link', 'version': '19.29.30153', 'commands': 'cl'}, 'cython': {'name': 'cython', 'linker': 'cython', 'version': '3.0.7', 'commands': 'cython'}, 'c++': {'name': 'msvc', 'linker': 'link', 'version': '19.29.30153', 'commands': 'cl'}}, 'Machine Information': {'host': {'cpu': 'x86_64', 'family': 'x86_64', 'endian': 'little', 'system': 'windows'}, 'build': {'cpu': 'x86_64', 'family': 'x86_64', 'endian': 'little', 'system': 'windows'}}, 'Build Dependencies': {'blas': {'name': 'openblas64', 'found': True, 'version': '0.3.23.dev', 'detection method': 'pkgconfig', 'include directory': '/c/opt/64/include', 'lib directory': '/c/opt/64/lib', 'openblas configuration': 'USE_64BITINT=1 DYNAMIC_ARCH=1 DYNAMIC_OLDER= NO_CBLAS= NO_LAPACK= NO_LAPACKE= NO_AFFINITY=1 USE_OPENMP= SKYLAKEX MAX_THREADS=2', 'pc file directory': 'C:/opt/64/lib/pkgconfig'}, 'lapack': {'name': 'dep2097330985504', 'found': True, 'version': '1.26.3', 'detection method': 'internal', 'include directory': 'unknown', 'lib directory': 'unknown', 'openblas configuration': 'unknown', 'pc file directory': 'unknown'}}, 'Python Information': {'path': 'C:\\Users\\runneradmin\\AppData\\Local\\Temp\\cibw-run-a2wx5bfb\\cp310-win_amd64\\build\\venv\\Scripts\\python.exe', 'version': '3.10'}, 'SIMD Extensions': {'baseline': ['SSE', 'SSE2', 'SSE3'], 'found': ['SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2'], 'not found': ['AVX512F', 'AVX512CD', 'AVX512_SKX', 'AVX512_CLX', 'AVX512_CNL', 'AVX512_ICL']}}
__cpu_baseline__: list = ['SSE', 'SSE2', 'SSE3']
__cpu_dispatch__: list = ['SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C', 'FMA3', 'AVX2', 'AVX512F', 'AVX512CD', 'AVX512_SKX', 'AVX512_CLX', 'AVX512_CNL', 'AVX512_ICL']
__cpu_features__: dict = {'MMX': True, 'SSE': True, 'SSE2': True, 'SSE3': True, 'SSSE3': True, 'SSE41': True, 'POPCNT': True, 'SSE42': True, 'AVX': True, 'F16C': True, 'XOP': False, 'FMA4': False, 'FMA3': True, 'AVX2': True, 'AVX512F': False, 'AVX512CD': False, 'AVX512ER': False, 'AVX512PF': False, 'AVX5124FMAPS': False, 'AVX5124VNNIW': False, 'AVX512VPOPCNTDQ': False, 'AVX512VL': False, 'AVX512BW': False, 'AVX512DQ': False, 'AVX512VNNI': False, 'AVX512IFMA': False, 'AVX512VBMI': False, 'AVX512VBMI2': False, 'AVX512BITALG': False, 'AVX512FP16': False, 'AVX512_KNL': False, 'AVX512_KNM': False, 'AVX512_SKX': False, 'AVX512_CLX': False, 'AVX512_CNL': False, 'AVX512_ICL': False, 'AVX512_SPR': False, 'VSX': False, 'VSX2': False, 'VSX3': False, 'VSX4': False, 'VX': False, 'VXE': False, 'VXE2': False, 'NEON': False, 'NEON_FP16': False, 'NEON_VFPV4': False, 'ASIMD': False, 'FPHP': False, 'ASIMDHP': False, 'ASIMDDP': False, 'ASIMDFHM': False}
_built_with_meson: bool = True
