
import binascii
import builtins
import codecs
import ctypes
import inspect
from codecs import encode
from collections import defaultdict
from functools import partial, wraps
from copy import copy
import numpy as np
from IPython.display import HTML, JSON
from IPython.utils.ipstruct import Struct as struct
from IPython.utils import io
from matplotlib.cbook import flatten
from IPython.core.interactiveshell import InteractiveShell as shell


def toggle(state = False):
    toggle.__defaults__, state =  (not state,), not state
    return state

def reveal(opt, input = None):
    shell.ast_node_interactivity = opt
    return input
def show():
    if toggle():
        reveal('all')
    else:
        reveal('none')
    return shell.ast_node_interactivity

def hide(): return show()

#Options: ‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’
def any(*x):
    _any = np.any if isinstance(x, np.ndarray) else builtins.any
    return _any(x[-1]) if len(x) == 1 else _any(x)

def all(*x):
    _all = np.all if isinstance(x, np.ndarray) else builtins.all
    return _all(x[-1]) if len(x) == 1 else _all(x)

class limit(int):
    def __call__(self, step=1, **kw):
        range = np.arange(0, self, step)
        if not any(['list' in kw, 'l' in kw]):
            return range
        return range.tolist()

class bounds(tuple):
    def __call__(self, step=1, **kw):
        range = (*self, step) if len(self) < 3 else self[:3]
        range = np.arange(*range)
        if not any(['list' in kw, 'l' in kw]):
            return range
        return range.tolist()


class bound(tuple):
    def __new__(self, *x, **kw):
        if len(x) == 1:
            return limit(x[0], **kw)
        return bounds(x, **kw)

class Map(struct):
    _current  = []
    _canaries = '_ipython_canary_method_should_not_exist_', '_ipython_display_', '_repr_html_', '__getstate__', '__wrapped__'
    _display  = "name", "_header",
    _h1   = """
    <h1 style = "
        font-size             : 14px;
        color                 : white;
        display               : block;
        margin                : 0 auto;
        width                 : fit-content;
        font-weight           : 600;
        font-family           : SFNS Display, system-ui !important;
        -webkit-font-smoothing: antialiased">
        {name}{view}
    </h1>"""

    @property
    def name(self):
        self._header   = partial(type(self)._h1.format, name = self['name'])
        display(HTML(self._header(view = "")))
        ret = lambda: self['name']
        with io.capture_output() as captured:
            return ret();

    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]

    def __init__(self, name="mapping", *dicts, **kw):
        for d in dicts:
            if isinstance(d, dict):
                self.update(d)
        self.update(kw)
        self.name = name
        h1 = partial(type(self)._h1.format, name = name)
        self._header = h1


    def __repr__(self):
        cleared = self.clear()
        out = dict()
        out.update(dict.items(self))
        for key, value in list(out.items()).copy():
            if key in type(self)._canaries + type(self)._display:
                    out.pop(key)
            elif isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, type(self)):
                out[key] = dict(value.clear())
            else:
                try:
                    hash(value)
                except Exception as e:
                    print(f"{key} of type {type(value)} unhashable. Converting to string")
                    out[key] = str(value)
        # self.display("", cleared)
        self.display("", out)
        return ""


    def __add__(self, other):
        if isinstance(self, type(other)):
            return type(self)({** self, ** other})
        return self

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def clear(self):
        out = dict()
        out.update(dict.items(self))
        for key, value in list(out.items()).copy():
            if key in type(self)._canaries + type(self)._display:
                out.pop(key)
            elif isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, type(self)):
                # out[key] = dict(value.clear())
                pass
            else:
                try:
                    hash(value)
                except Exception as e:
                    print(f"{key} of type {type(value)} unhashable. Converting to string")
                    out[key] = str(value)
        return out

    def display(self, header, cleared):
        # self._header = partial(type(self)._h1.format, name = self.name)
        self.name;
        # display(HTML(self._header(view = header)))
        reveal('last')
        try:
            display(JSON(cleared))
        except Exception as e:
            print(e, f"Map {self.name} is not JSON-able.")
            from pprint import pprint
            pprint(cleared)


    def values(self):
        cleared = list(self.clear().values())
        for i, value in enumerate(cleared):
            if isinstance(value, type(self)):
                cleared[i] = (key, value.clear())
        self.display(": values", cleared)
        try:
            return np.array(cleared)
        except Exception as e:
            return cleared

    def keys(self):
        cleared = list(self.clear().keys())
        self.display(": keys", cleared)
        return cleared

    def items(self):
        keys = self.keys();
        values = self.values();
        cleared = list(zip(keys, values))
        self.display("", cleared)
        return cleared

    @classmethod
    def create(cls, num = None, *names):
        if isinstance(num, str):
            names, num = [num, * names], None
        if not any([num, names]):
            cls._current.append(cls())
            return cls._current[-1]
        if names:
            maps = [cls(name) for name in names]
        elif num:
            maps = [cls(num) for num in bound(num)()]
        if len(maps) == 1:
            return maps[0]
        return maps

    def copy(self):
        cls = type(self)
        return cls(self.name, super().copy())

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


PyObject._fields_ = [("ob_refcnt", Py_ssize_t),
                     ("ob_type", ctypes.POINTER(PyObject))]


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
