module Data.Array.Accelerate.BLAS.Internal.Dot where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA hiding (create)
import Data.Array.Accelerate hiding (snd)
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith)
--import System.Mem.Weak (addFinalizer)
-- import System.IO.Unsafe (unsafePerformIO)
import qualified Foreign.CUDA as C
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
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

cudaDotProductF :: (Vector CFloat, Vector CFloat) -> CIO (Scalar CFloat)
cudaDotProductF (v1,v2) = do
  traceShowM $ "cudaDotProductF"
  let n = arraySize (arrayShape v1)
  traceShowM $ n
  --o <- allocateArray Z
  -- need "handle" and pointer to v1 and "stride between consecutive elements of v1" and
  -- pointer to v2 and "stride between consecutive elements of v2" and result output array

  ((),v1ptr) <- devicePtrsOfArray v1
  ((),v2ptr) <- devicePtrsOfArray v2
  --((),optr)  <- devicePtrsOfArray o
  
  o <- liftIO $ BL.dot theHandle n v1ptr 1 v2ptr 1

  return $ fromList Z [o]

sdot :: Acc (Vector CFloat) -> Acc (Vector CFloat) -> Acc (Scalar CFloat)
sdot v1 v2 = foreignAcc cudaDot pureDot $ lift (v1,v2)
  where cudaDot = CUDAForeignAcc "cudaDotProductF" (\stream -> cudaDotProductF)
        pureDot :: Acc (Vector CFloat, Vector CFloat) -> Acc (Scalar CFloat)
        pureDot vs = let (u,v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v

test = do
  let x = fromList (Z:.10) [1..10] :: Vector CFloat
  let y = fromList (Z:.10) [2..11] :: Vector CFloat

  run $ sdot (use x) (use y)
