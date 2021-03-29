# ebmrPy
## Empirical Bayes Multiple Regression (EBMR)
This is a Python version of the EBMR package ([ebmr.alpha](https://github.com/stephenslab/ebmr.alpha)) by [@stephens999](https://github.com/stephens999). 
The theory was developed by Matthew Stephens: [Link to Overleaf](https://www.overleaf.com/project/5efcaef995af9b00012d0576)

### Installation
For development, download this repository and install using `pip`:
```
git clone https://github.com/banskt/ebmrPy.git # or use the SSH link
cd ebmrPy
pip install -e .
```

### How to use
Functions are not documented yet. Here is only a quick start.
```
from ebmrPy.inference.ebmr import EBMR
ebmr = EBMR(X, y, prior = 'mix_point', 
            grr = 'em_svd', sigma = 'full', inverse = 'direct',
            s2_init = 1, sb2_init = 1,
            max_iter = 100, tol = 1e-8,
            mll_calc = True,
            mix_point_w = np.array([0.001, 1.0, 2.0, 3.0, 4.0]),
            ignore_convergence = True
           )
ebmr.update()
```

### Running tests
Run the unittest from the `/path/to/download/ebmrPy` directory.
```
python -m unittest
```
