module Data.Array.Accelerate.BLAS.Internal.Gemm where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA hiding (create)
import Data.Array.Accelerate hiding (snd)
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith)
import qualified Foreign.CUDA as C
import Foreign.CUDA.Ptr
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
import Debug.Trace


cudaGemmF :: Maybe Stream -> (Array DIM2 Float, Array DIM2 Float) -> CIO (Array DIM2 Float)
cudaGemmF ms (a,b) = do
  traceShowM "cudaGemmF"
  let Z :. ra :. ca = arrayShape a
      Z :. rb :. cb  = arrayShape b
  traceShowM (ra,ca,rb,cb)
  c <- allocateArray $ Z :. ra :. cb
  withDevicePtrs a ms $ \aptr -> do
    withDevicePtrs b ms $ \bptr -> do
      withDevicePtrs c ms $ \cptr -> do
        liftIO $ BL.gemm theHandle BL.N BL.N ra cb ca (CFloat 1) (castDevPtr aptr) ra (castDevPtr bptr) rb (CFloat 0) (castDevPtr cptr) ra
        return c

--sdot :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
--sdot v1 v2 = foreignAcc cudaDot pureDot $ lift (v1,v2)
--  where cudaDot = CUDAForeignAcc "cudaDotProductF" (\stream -> cudaDotProductF (Just stream))
--        pureDot :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
--        pureDot vs = let (u,v) = unlift vs
--                     in fold (+) 0 $ zipWith (*) u v

--test = do
--  let x = fromList (Z:.10) [1..10] :: Vector Float
--  let y = fromList (Z:.10) [2..11] :: Vector Float

--  run $ sdot (use x) (use y)
