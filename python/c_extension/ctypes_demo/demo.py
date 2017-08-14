import ctypes
import os
import sys

_file = '_demo.so'
_path = os.path.join(sys.path[0], _file)
_mod = ctypes.cdll.LoadLibrary(_path)

# int gcd(int, int)
gcd = _mod.gcd
gcd.argtypes = (ctypes.c_int, ctypes.c_int)
gcd.restype = ctypes.c_int

print(gcd(1, 2))
