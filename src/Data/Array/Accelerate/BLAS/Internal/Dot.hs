module Data.Array.Accelerate.BLAS.Internal.Dot where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith)
import Foreign.CUDA.Ptr
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
-- import Debug.Trace

cudaDotProductF :: Maybe Stream -> (Vector Float, Vector Float) -> CIO (Scalar Float)
cudaDotProductF ms (v1,v2) = do
  -- traceShowM "cudaDotProductF"
  let n = arraySize (arrayShape v1)
  -- traceShowM n

  withDevicePtrs v1 ms $ \v1ptr -> do
    withDevicePtrs v2 ms $ \v2ptr -> do
      case ms of
        Just s -> do
          -- traceShowM "Just"
          liftIO $ BLF.setStream theHandle s
          CFloat o <- liftIO $ BL.dot theHandle n (castDevPtr v1ptr) 1 (castDevPtr v2ptr) 1
          -- traceShowM o
          let oarr = fromList Z [o]
          -- traceShowM oarr
          useArray oarr
          -- traceShowM oarr
          return oarr
        Nothing -> do
          CFloat o <- liftIO $ BL.dot theHandle n (castDevPtr v1ptr) 1 (castDevPtr v2ptr) 1
          -- traceShowM o
          let oarr = fromList Z [o]
          -- traceShowM oarr
          useArray oarr
          -- traceShowM oarr
          return oarr

sdot :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
sdot v1 v2 = foreignAcc cudaDot pureDot $ lift (v1,v2)
  where cudaDot = CUDAForeignAcc "cudaDotProductF" (\stream -> cudaDotProductF (Just stream))
        pureDot :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
        pureDot vs = let (u,v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v

test = do
  let x = fromList (Z:.10) [1..10] :: Vector Float
  let y = fromList (Z:.10) [2..11] :: Vector Float

  run $ sdot (use x) (use y)
