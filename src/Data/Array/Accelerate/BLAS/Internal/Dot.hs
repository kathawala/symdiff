module Data.Array.Accelerate.BLAS.Internal.Dot where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA hiding (create)
import Data.Array.Accelerate hiding (snd)
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith)
--import System.Mem.Weak (addFinalizer)
-- import System.IO.Unsafe (unsafePerformIO)
import Foreign.CUDA as C
import Debug.Trace

-- D.initialise []
-- dev <- device 0
-- ctx <- C.create dev []


--handle :: Handle
--handle = unsafePerformIO $ do
--  h <- create
--  putStrLn "Initialization Complete (GPU Device found)"
--  --addFinalizer handle (destroy h >> putStrLn "GPU Device work complete")
--  return h

cudaDotProductF :: Handle -> (Vector Float, Vector Float) -> CIO (Scalar Float)
cudaDotProductF handle (v1,v2) = do
  traceShowM $ "cudaDotProductF"
  let n = arraySize (arrayShape v1)
  traceShowM $ n
  o <- allocateArray Z
  -- need "handle" and pointer to v1 and "stride between consecutive elements of v1" and
  -- pointer to v2 and "stride between consecutive elements of v2" and result output array

  ((),v1ptr) <- devicePtrsOfArray v1
  ((),v2ptr) <- devicePtrsOfArray v2
  ((),optr)  <- devicePtrsOfArray o
  
  traceShowM $ handle
  traceShowM $ "got handle, calling cublasSdot"
  liftIO $ cublasSdot handle n v1ptr 1 v2ptr 1 optr

  return o

sdot :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
sdot v1 v2 = foreignAcc cudaDot pureDot $ lift (v1,v2)
  where cudaDot = CUDAForeignAcc "cudaDotProductF" (\stream -> cudaDotProductF theHandle)
        pureDot :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
        pureDot vs = let (u,v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v

test = do
  let x = fromList (Z:.10) [1..10] :: Vector Float
  let y = fromList (Z:.10) [2..11] :: Vector Float

  run $ sdot (use x) (use y)
