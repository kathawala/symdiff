module Examples.NeuralNet where

import Data.Array.Accelerate.BLAS.Internal.Dot
import Data.Array.Accelerate.BLAS.Internal.Gemm
import Data.Array.Accelerate.BLAS.Internal.Axpy
import Data.Array.Accelerate hiding (length)
import Prelude hiding (map, replicate)
import Data.Array.Accelerate.Interpreter as I
-- import Data.Array.Accelerate.CUDA as I

-- example: toMatrix [1,2,3,4] (2,2) gives back:
       -- [1  2]
       -- [3  4]
toMatrix :: [Float] -> (Int,Int) -> Acc (Matrix Float)
toMatrix x (rx,cx)           = use (fromList (Z :. rx :. cx) x)

toVector :: [Float] -> Acc (Vector Float)
toVector x                   = use (fromList (Z :. (length x)) x)

clevel :: Acc (Vector Float) -> Acc (Matrix Float) -> Acc (Vector Float) -> Acc (Vector Float)
clevel x w b  = axpy (gemv w x) b
  -- where 
    -- x          = replicate (lift (Z :. All :. (1::Int))) (x0)
    -- wTimesX    = slice (gemm w x) (lift (Z :. All :. (0::Int)))

hlevel :: Acc (Vector Float) -> Acc (Vector Float)
hlevel = map tanh

-- train is used as follows
-- train numTrainingCycles featureVec weightMatrix penaltyVec
-- vectors and matrices must be initialized using toMatrix and toVector
train :: Int -> Acc (Vector Float) -> Acc (Matrix Float) -> Acc (Vector Float) -> Acc (Vector Float)
train n x w b
  | n == 0     = x 
  | otherwise  = train (n-1) (hlevel $ clevel x w b) w b











--mlp x (w1, b1, w2, b2) = layer w2 b2 $ layer w1 b1 x where layer w b h = map tanh $ axpy b $ gemv w h

--layer x (w, b) = map tanh $ axpy b $ gemv w x

--diff (layer x) dy (w, b) = (dw, db) where
--    dc = map tanh' dy
--    da = dc
--    db = dc
--    dx = gevm da w
--    dw = gevv da x

--layer x (w, b) = map tanh $ axpy b $ gemv w x
--diff (layer x) dy (w, b) =
--	dc = map tanh' dy

