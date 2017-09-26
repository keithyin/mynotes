import ctypes
import os
import sys

_file = '_demo.so'
_path = os.path.join(sys.path[0], _file)
_mod = ctypes.cdll.LoadLibrary(_path)

# int gcd(int, int)
gcd = _mod.gcd
# gcd.argtypes = (ctypes.c_int, ctypes.c_int)
# gcd.restype = ctypes.c_int

print(gcd(3, 4))


_divide = _mod.divide
_divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_divide.restype = ctypes.c_int


def divide(a, b):
    rem = ctypes.c_int()
    quot = _divide(a, b, rem)
    print(quot, rem.value)


class DoubleArrayType(object):
    # entrance
    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise ValueError('Type Error')

    def from_list(self, param):
        # ((ctypes.c_double * len(param)) create a new type and init it.
        val = ((ctypes.c_double * len(param))(*param))
        return val

    from_tuple = from_list

    def from_ndarray(self, param):
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


DoubleArray = DoubleArrayType()
_avg = _mod.avg
_avg.argtypes = (DoubleArray, ctypes.c_int)
_avg.restype = ctypes.c_double


def avg(values):
    print(_avg(values, len(values)))


avg([1, 3, 4, 5])


class Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]


distance = _mod.distance
distance.argtypes = (ctypes.POINTER(Point), ctypes.POINTER(Point))
distance.restype = ctypes.c_double

res = distance(Point(1, 2), Point(2, 3))

print(res)
new_distance = _mod.new_distance
new_distance.argtypes = (ctypes.POINTER(Point), ctypes.POINTER(Point))
new_distance.restype = Point

print(new_distance(Point(1, 2), Point(3, 4)).y)
