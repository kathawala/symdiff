# symdiff
GPU-powered symbolic differentiation in haskell

# Status

Proof of concept generation of gradient expressions is in the `test` function in `Diff.hs`

##Build Steps
Before building, you will need to have CUDA installed on your system, both the developer driver and SDK.

You will also need the git versions of the [accelerate library](https://github.com/AccelerateHS/accelerate) and the
[accelerate-cuda library](https://github.com/AccelerateHS/accelerate-cuda).
These versions are not available on hackage, so you will need to clone the repo and build it yourself.

Assuming you have CUDA and `nvcc` in your `PATH`, then you can download and build the library as follows:
```
git clone https://github.com/AccelerateHS/accelerate.git
git clone https://github.com/AccelerateHS/accelerate-cuda.git
git clone https://github.com/kathawala/symdiff.git
cd symdiff
cabal sandbox init
cabal sandbox add-source ../accelerate
cabal sandbox add-source ../accelerate-cuda
cabal sandbox add-source ./cublas
cabal install
```
