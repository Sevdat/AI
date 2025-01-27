"""

Build a c-extension module on-the-fly in tests.
See build_and_import_extensions for usage hints

"""
from __future__ import annotations
import os as os
import pathlib as pathlib
import subprocess as subprocess
import sys as sys
import sysconfig as sysconfig
import textwrap as textwrap
__all__: list = ['build_and_import_extension', 'compile_extension_module']
def _c_compile(cfile, outputfilename, include_dirs = list(), libraries = list(), library_dirs = list()):
    ...
def _convert_str_to_file(source, dirname):
    """
    Helper function to create a file ``source.c`` in `dirname` that contains
        the string in `source`. Returns the file name
        
    """
def _make_methods(functions, modname):
    """
     Turns the name, signature, code in functions into complete functions
        and lists them in a methods_table. Then turns the methods_table into a
        ``PyMethodDef`` structure and returns the resulting code fragment ready
        for compilation
        
    """
def _make_source(name, init, body):
    """
     Combines the code fragments into source code ready to be compiled
        
    """
def build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs):
    """
    use meson to build
    """
def build_and_import_extension(modname, functions, *, prologue = '', build_dir = None, include_dirs = list(), more_init = ''):
    """
    
        Build and imports a c-extension module `modname` from a list of function
        fragments `functions`.
    
    
        Parameters
        ----------
        functions : list of fragments
            Each fragment is a sequence of func_name, calling convention, snippet.
        prologue : string
            Code to precede the rest, usually extra ``#include`` or ``#define``
            macros.
        build_dir : pathlib.Path
            Where to build the module, usually a temporary directory
        include_dirs : list
            Extra directories to find include files when compiling
        more_init : string
            Code to appear in the module PyMODINIT_FUNC
    
        Returns
        -------
        out: module
            The module will have been loaded and is ready for use
    
        Examples
        --------
        >>> functions = [("test_bytes", "METH_O", \"\"\"
            if ( !PyBytesCheck(args)) {
                Py_RETURN_FALSE;
            }
            Py_RETURN_TRUE;
        \"\"\")]
        >>> mod = build_and_import_extension("testme", functions)
        >>> assert not mod.test_bytes(u'abc')
        >>> assert mod.test_bytes(b'abc')
        
    """
def compile_extension_module(name, builddir, include_dirs, source_string, libraries = list(), library_dirs = list()):
    """
    
        Build an extension module and return the filename of the resulting
        native code file.
    
        Parameters
        ----------
        name : string
            name of the module, possibly including dots if it is a module inside a
            package.
        builddir : pathlib.Path
            Where to build the module, usually a temporary directory
        include_dirs : list
            Extra directories to find include files when compiling
        libraries : list
            Libraries to link into the extension module
        library_dirs: list
            Where to find the libraries, ``-L`` passed to the linker
        
    """
def get_so_suffix():
    ...
