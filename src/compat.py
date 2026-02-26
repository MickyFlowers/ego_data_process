"""
Python 3.11+ compatibility: inspect.getargspec was removed, alias to getfullargspec.
Import this module first (e.g. in process entry points) before any lib that uses getargspec.
"""
import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
