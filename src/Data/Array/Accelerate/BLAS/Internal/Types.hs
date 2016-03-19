module Data.Array.Accelerate.BLAS.Internal.Types where

import Data.Array.Accelerate

type Matrix a = Array DIM2 a

type Scal = Acc (Scalar Float)
type Vect = Acc (Vector Float)
type Matr = Acc (Matrix Float)
