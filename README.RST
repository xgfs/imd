===============================
Intrinsic Multi-scale Evaluation of Generative Models
===============================

This repository provides a reference implementation of MSID, a metric for comparing underlying *intrinsic* geometry of data manifolds (`paper <https://arxiv.org/abs/1905.11141>`_).

Installation
-----------

Installation is as simple as ``python setup.py install``.

Requirements
-----------

* Python 2.7 or Python 3.3+
* `SciPy <http://www.scipy.org/install.html/>`_
* `NumPy <http://www.numpy.org/>`_
* [optional] `pykgraph <https://github.com/aaalgo/kgraph/>`_, for Anaconda users, just ``conda install pykgraph``

Example usage
-----------

.. code-block:: python

    import numpy as np
    from msid import msid_score

    np.random.seed(1)

    x0 = np.random.randn(1000, 10)
    x1 = np.random.randn(1000, 9) # MSID can compare two data distributions with different dimensionalities
    y0 = np.random.beta(0.5, 0.5, (1000, 10))

    print('x0=N(0, 1), shape=', x0.shape)
    print('x1=N(0, 1), shape=', x1.shape)
    print('y0=beta(0.5, 0.5), shape=', y0.shape)

    print('MSID(x0, x1)', msid_score(x0, x1))
    print('MSID(x0, y0)', msid_score(x0, y0))

Contact
-----------

``echo "%7=87@=<2=<>5.27" | tr '#-)/->' '_-|'``
