**Codes used for a talk "Python for HPC in CFD"**

**Abstract**
The Computational Fluid Dynamics (CFD) simulations for industrial problems require substantial computational resources to get accurate results in a reasonable time. Therefore, due to their efficiency, most commercial and academic CFD codes are developed using C/C++ or Fortran. However, creating an optimized CFD solver using Fortran or C/C++ takes significant time as learning C++ is more challenging than Python because of complex syntax, memory management, package management, etc. Whereas Python is a versatile language, easy to learn, read, write and understand. It has many third-party libraries, so it is easy to prototype ideas in Python. It is the most popular among programming languages chosen to teach university-level introductory computer science courses. Because of all these and many other features, it is widely used in various industries. But Python isn't commonly used in CFD because pure Python is very slow for number crunching. However, for scientific computing, we do not use pure Python; instead, we use it as a layer on top of efficient compiled packages written in C/C++, Fortran, etc. The most popular package used in the scientific community is undoubtedly NumPy. And codes written in NumPy are much faster than pure Python code. Although codes written using NumPy are faster, using for-loops in NumPy is inefficient. Therefore, Python-NumPy code should not contain explicit for-loops to achieve nearC/C++ speed. Fortunately, most CFD codes can be written without explicit for-loops, but there will be situations where we can't avoid for-loops, and to improve the performance in these situations, many solutions exist in the Python ecosystem. This talk focuses on solutions based on Just-In-Time (JIT) compilers like Numba and Jax, as these offer solutions with no or minimal code change. JIT compilers translate Python functions to optimized machine code at runtime. The JIT compilation results in C/C++-like speeds in many cases. Another significant advantage of JIT compilers is they can generate efficient machine code for accelerators like GPUs with no or minimal code change. This talk will teach us how to develop efficient CFD codes to solve 1D compressible Euler equations using vectorized NumPy, Numba 

This repository contains C++, Python(NumPy with for-loop, vectorized, numba and jax) codes
for solving one-dimensional compressible Euler equations using the Finite-Volume Method (FVM)

**Compiling and running C++ code**

*Compile*
```
cd cpp 
g++ euler_1d.cpp -std=c++17
```

*Run*
```
./a.out
```

*Plot*
```
jupyter-lab plot_results.ipynb
```




