"""
Dumb little utility for adding the top-level resonator_networks directory
to the python module path at runtime. Should be sufficient for any script to
just call ``import _set_the_path``. There are probably better ways to do this.
"""
import sys
import os
examples_fullpath = os.path.dirname(os.path.abspath(__file__))
toplevel_dir_fullpath = examples_fullpath[:examples_fullpath.rfind('/')+1]
sys.path.insert(0, toplevel_dir_fullpath)
