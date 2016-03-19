module Examples.NeuralNet where

import Data.Array.Accelerate.BLAS.Internal.Dot
import Data.Array.Accelerate.BLAS.Internal.Gemm
import Data.Array.Accelerate.BLAS.Internal.Axpy
import Data.Array.Accelerate.BLAS.Internal.Types
import Data.Array.Accelerate hiding (length)
import Prelude hiding (map, replicate)
-- import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.CUDA as I

-- example: toMatrix [1,2,3,4] (2,2) gives back:
       -- [1  2]
       -- [3  4]
toMatrix :: [Float] -> (Int,Int) -> Matr
toMatrix x (rx,cx)           = use (fromList (Z :. rx :. cx) x)

toVector :: [Float] -> Vect
toVector x                   = use (fromList (Z :. (length x)) x)

mlp :: Matr -> Vect -> Matr -> Vect -> Vect -> Vect
mlp w1 b1 w2 b2 x = layer w2 b2 $ layer w1 b1 x
  where layer w b x = map tanh $ axpy (b,(gemv (w,x)))
  