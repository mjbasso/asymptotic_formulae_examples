Asymptotic Formulae Examples
============================

Python 3 code for generating the data and producing the plots included in "Generalized asymptotic formulae for estimating statistical significance in high energy physics analyses" ([arXiv:2102.04275](https://arxiv.org/abs/2102.04275)).

Installing
----------

To install the necessary Python environment, run:

```bash
source install.sh
```

Running:
--------

To produce the plots from the paper (N.B.: from scratch, this will take a *long* time), run:

```bash
python3 make_paper_plots.py
```

In Python programs, the functions from `asymptotic_formulae` can be imported and used like:

```python
from asymptotic_formulae import nSRZ0

s   = [40., 10., 12.]
b   = 1000.
tau = [2., 10., 20.]
z0  = nSRZ0(s, b, tau))
```

Returning:
----------

To set up the Python environment upon returning to the directory, run:

```bash
source setup.sh
```
