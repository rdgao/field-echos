
from codecs import encode
from functools import wraps
from matplotlib.cbook import flatten
from collections import defaultdict
from IPython.utils.ipstruct import Struct as struct
import binascii
import builtins
import codecs
import ctypes
import inspect
import numpy as np


class mapping(struct):
    canaries = '_ipython_canary_method_should_not_exist_', '_ipython_display_', '_repr_html_'

    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]

    def __repr__(self):
        # from json import dumps
        # from IPython.display import JSON
        # from copy import deepcopy as copy
        self.clear()
        # display(JSON(dumps(copy(self))))
        return str("")


    def clear(self):
        for key in type(self).canaries:
            if key in self:
                self.pop(key)

    def __add__(self, other):
        if isinstance(self, type(other)):
            return type(self)({** self, ** other})
        return self

    def __iadd__(self, other):
            self = self.__add__(other)
            return self

    # def __getattribute__(self, key):
    #     try:
    #         ret = self.__getitem__(key) if self.__getitem__(key) else None
    #         if ret:
    #             return ret
    #     except: pass
    #     else:
    #         return super().__getattribute__()

current = []
def mappings(num = False):
    if not num:
        return current
    else:
        ret = []
        while(num):
            ret.append(mapping())
            num = num - 1
        current.extend(ret);
        return ret


class limit(int):
    def __call__(self, step = 1):
        return np.arange(0,self, step)

class bounds(tuple):
    def __call__(self, step = 1):
        range = (*self, step) if len(self) < 3 else self[:3]
        return np.arange(*range)

class bound(tuple):
    def __new__(self, *x):
        if len(x) == 1:
            return limit(x[0])
        return bounds(x)


def flat(t): return np.array(list(flatten(t)))

class shortString(str):

    def __init__(self):
        super()
    def _hex_repr_(self):
        return
    def __call__(self): pass

Py_ssize_t = (
    hasattr(ctypes.pythonapi, "Py_InitModule4_64") and ctypes.c_int64 or ctypes.c_int
)


class PyObject(ctypes.Structure):
    pass


PyObject._fields_ = [("ob_refcnt", Py_ssize_t), ("ob_type", ctypes.POINTER(PyObject))]


class SlotsProxy(PyObject):
    _fields_ = [("dict", ctypes.POINTER(PyObject))]


def injectableBuiltin(cls):
    # It's important to create variables here, we want those objects alive
    # within this whole scope.
    name = cls.__name__ if hasattr(cls, "__name__") else "key"
    try:
        target = cls.__dict__
    except:
        "Probably a tuple."
    # Introspection to find the `PyProxyDict` object that contains the
    # precious `dict` attribute.
    proxy_dict = SlotsProxy.from_address(id(target))
    namespace = {}

    # `Cast` the `proxy_dict.dict` into a python objectself.
    # The `from_address()` function returns the `py_object` version.
    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace), ctypes.py_object(name), proxy_dict.dict
    )
    return namespace[name]

def inject(cls, attr, value, hide_from_dir=False):
    """Curse a built-in `cls` with `attr` set to `value`

    This function monkey-injects the built-in python object `attr` adding a new
    attribute to it. You can add any kind of argument to the `class`.

    It's possible to attach methods as class methods, just do the following:

      >>> def myclassmethod(cls):
      ...     return cls(1.5)
      >>> inject(float, "myclassmethod", classmethod(myclassmethod))
      >>> float.myclassmethod()
      1.5

    Methods will be automatically bound, so don't forget to add a self
    parameter to them, like this:

      >>> def hello(self):
      ...     return self * 2
      >>> inject(str, "hello", hello)
      >>> "yo".hello()
      "yoyo"
    """
    dikt = injectableBuiltin(cls)

    old_value = dikt.get(attr, None)
    oldName = "_c_%s" % attr  # do not use .format here, it breaks py2.{5,6}
    if old_value:
        dikt[oldName] = old_value

    if old_value:
        dikt[attr] = value

        try:
            dikt[attr].__name__ = old_value.__name__
        except (AttributeError, TypeError):  # py2.5 will raise `TypeError`
            pass
        try:
            dikt[attr].__qualname__ = old_value.__qualname__
        except AttributeError:
            pass
    else:
        dikt[attr] = value

    if hide_from_dir:
        __hidden_elements__[cls.__name__].append(attr)


def reverse(cls, attr):
    """Reverse a inject in a built-in object- ! removes any attribute from any built-in class.

    Good:

      >>> inject(str, "blah", "bleh")
      >>> assert "blah" in dir(str)
      >>> reverse(str, "blah")
      >>> assert "blah" not in dir(str)

    Bad:

      >>> reverse(str, "strip")
      >>> " blah ".strip()
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      AttributeError: 'str' object has no attribute 'strip'

    """
    dikt = injectableBuiltin(cls)
    del dikt[attr]

def injection(cls, name):
    """Decorator to add decorated method named `name` the class `cls`

    So you can use it like this:

        >>> @injection(dict, 'banner')
        ... def dict_banner(self):
        ...     l = len(self)
        ...     print('This dict has {0} element{1}'.format(
        ...         l, l is 1 and '' or 's')
        >>> {'a': 1, 'b': 2}.banner()
        'This dict has 2 elements'
    """

    def wrapper(func):
        inject(cls, name, func)
        return func

    return wrapper




@injection(list, "join")
def join(self, sep=None, *a, **k):
    if not sep or sep in (None, False, ""):
        return "".join(list(self), *a, **k)
    else:
        return sep.join(list(self), *a, **k)

# from IPython.core.display import HTML
# import urllib.request
# url = 'https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/kf_book/custom.css'
# response = urllib.request.urlopen(url)
# HTML(response.read().decode("utf-8"))
