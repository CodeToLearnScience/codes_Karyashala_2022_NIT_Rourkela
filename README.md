**Codes used for a talk "Python for HPC in CFD"**

This repository contains C++, Python(NumPy with for-loop, vectorized, numba and jax unoptimised) codes
for solving one-dimensional compressible Euler equations using the Finite-Volume Method (FVM)

**Compiling and running C++ code**

*Compile*
```
cd cpp 
g++ g++ euler_1d.cpp -std=c++17
```

*Run*
```
./a.out
```

*Plot*
```
jupyter-lab plot_results.ipynb
```




