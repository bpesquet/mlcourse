---
marp: true
---

<!-- Apply header and footer to first slide only -->
<!-- _header: "[![Bordeaux INP logo](../ensc_logo.jpg)](https://www.bordeaux-inp.fr)" -->
<!-- _footer: "[Baptiste Pesquet](https://www.bpesquet.fr)" -->

# Linear Regression

> Soon!

---

## Implementation details

The [code example](../../mlcourse/test_linear_regression.py) uses the PyTorch [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class to implement linear regression on a simple 2D dataset.

After gradients computation, parameters are updated manually to better illustrate how gradient descent works. Subsequent examples will use a predefined optimizer for concision.
