# symdiff
GPU-powered symbolic differentiation in haskell

##Build Steps
Before building, you will need to have CUDA installed on your system, both the developer driver and SDK.

You will also need the newest version (0.3.0.0) of the [cublas library](https://github.com/bmsherman/cublas).
This version is not available on hackage, so you will need to clone the repo and build it yourself.

Assuming you have CUDA and `nvcc` in your PATH, then you can download and build the library as follows:
```
git clone https://github.com/bmsherman/cublas.git
mv cublas cublas-0.3.0.0
git clone https://github.com/kathawala/symdiff.git
cd symdiff
cabal sandbox init
cabal sandbox add-source ../cublas-0.3.0.0
cabal install
```
