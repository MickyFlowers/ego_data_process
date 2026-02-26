"""
Compatibility patches. Import this module first in process entry points.

- Python 3.11+: inspect.getargspec was removed, alias to getfullargspec.
- NumPy 2.0+: np.int, np.float, np.bool, etc. were removed; some deps still do "from numpy import int".
"""
import builtins
import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# NumPy 2.0 removed deprecated type aliases; restore for deps that do "from numpy import int" etc.
import numpy as np

for _name in ("int", "float", "bool", "str", "complex", "object"):
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name))
# Python 3 has no unicode type (str is unicode); deps may still do "from numpy import unicode"
if not hasattr(np, "unicode"):
    np.unicode = str
