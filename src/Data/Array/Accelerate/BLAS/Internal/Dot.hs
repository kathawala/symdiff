module Data.Array.Accelerate.Symdiff where

import Data.Array.Accelerate hiding (snd)
import Data.Array.Accelerate.CUDA.Foreign
import qualified Foreign.CUDA.Cublas as BL
import Prelude hiding (zipWith)
import System.Mem.Weak (addFinalizer)
import System.IO.Unsafe (unsafePerformIO)
import Foreign.CUDA.Driver as D
import Foreign.CUDA as C

-- D.initialise []
-- dev <- device 0
-- ctx <- C.create dev []


handle :: BL.Handle
handle = unsafePerformIO $ do
  h <- BL.create
  putStrLn "Initialization Complete (GPU Device found)"
  addFinalizer handle (BL.destroy h >> putStrLn "GPU Device work complete")
  return h

cudaDotProductF :: (Vector Float, Vector Float) -> CIO (Scalar Float)
cudaDotProductF (v1,v2) = do
  let n = arraySize (arrayShape v1)

  o <- allocateArray Z
  -- need "handle" and pointer to v1 and "stride between consecutive elements of v1" and
  -- pointer to v2 and "stride between consecutive elements of v2" and result output array

  ((),v1ptr) <- devicePtrsOfArray v1
  ((),v2ptr) <- devicePtrsOfArray v2
  ((),optr)  <- devicePtrsOfArray o
  
  result <- liftIO $ BL.dot handle n v1ptr 1 v2ptr 1

  return result

sdot :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
sdot v1 v2 = foreignAcc cudaDot pureDot $ lift (v1,v2)
  where cudaDot = CUDAForeignAcc "cudaDotProductF" cudaDotProductF
        pureDot :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
        pureDot vs = let (u,v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v