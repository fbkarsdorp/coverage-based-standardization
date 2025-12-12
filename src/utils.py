import inspect
import numpy as np
import numbers


def get_arguments():
    """Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
    """
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args = dict(args)
    args.update(args.pop(kwname, []))
    if "self" in args:
        del args["self"]
    if "__class__" in args:
        del args["__class__"]
    return args


def reindex_array(x):
    _, x = np.unique(x, return_inverse=True)
    return x


def check_random_state(seed):
    if seed is np.random:
        return np.random.mtrand._rand
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def check_array(x):
    x = np.asarray(x)
    return np.array([x]) if x.ndim == 1 else x


def bincount2d(a):
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.flatten(), minlength=a.shape[0] * N).reshape(-1, N)


def hill_number(x, q, p):
    return (
        (x > 0).sum()
        if q == 0
        else np.exp(-np.sum(p * np.log(p)))
        if q == 1
        else np.exp(1 / (1 - q) * np.log(np.sum(p ** q)))
    )
