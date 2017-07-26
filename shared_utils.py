from multiprocessing import Value, RawArray
import ctypes
import numpy as np


class SharedCounter(object):
    def __init__(self, init_val=0):
        self.counter = Value(ctypes.c_int32, init_val)

    def progress(self, steps):
        with self.counter.get_lock():
            self.counter.value += steps

    def value(self):
        with self.counter.get_lock():
            return self.counter.value


class SharedVars(object):
    def __init__(self, num_actions, opt_type=None, lr=0):
        self.var_shapes = [
            (8, 8, 4, 32),
            (32),
            (4, 4, 32, 64),
            (64),
            (3, 3, 64, 64),
            (64),
            (3136, 512),  # 3136 = 7 * 7 * 64
            (512),
            (512, num_actions),
            (num_actions),
            (512, 1),
            (1)]
        # sum is a built-in function
        self.size = sum([np.prod(shape) for shape in self.var_shapes])

        if opt_type == 'adam':
            self.ms = self.malloc_contiguous(self.size)
            self.vs = self.malloc_contiguous(self.size)
            self.lr = RawValue(ctypes.c_float, lr)
        elif opt_type == 'rmsprop':
            self.vars = self.malloc_contiguous(self.size, np.ones(self.size, dtype=np.float))
        elif opt_type == 'momentum':
            self.vars = self.malloc_contiguous(self.size)
        else:
            self.vars = self.malloc_contiguous(self.size)

    def malloc_contiguous(self, size, init_val=None):
        if init_val is None:
            size = int(size)
            return RawArray(ctypes.c_float, size)
        else:
            return RawArray(ctypes.c_float, init_val)
