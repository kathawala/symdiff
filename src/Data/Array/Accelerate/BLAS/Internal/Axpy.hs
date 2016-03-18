module Data.Array.Accelerate.BLAS.Internal.Axpy where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.BLAS.Internal.Types
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith)
import Foreign.CUDA.Ptr
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
-- import Debug.Trace

cudaAxpyF :: Maybe Stream -> (Vector Float, Vector Float) -> CIO (Vector Float)
cudaAxpyF ms (v1,v2) = do
  let n  = arraySize (arrayShape v1)
  v3 <- allocateArray $ Z :. n :: CIO (Vector Float)



  withDevicePtrs v1 ms $ \v1ptr -> do
    withDevicePtrs v2 ms $ \v2ptr -> do
      withDevicePtrs v3 ms $ \v3ptr -> do
        -- to maintain data integrity, v3 is just a copy of v2
        liftIO $ BL.copy theHandle n (castDevPtr v2ptr) 1 ((castDevPtr v3ptr) :: DevicePtr CFloat) 1
        -- here the value of v3 is overwritten with the result of addition
        liftIO $ BL.axpy theHandle n (CFloat 1) (castDevPtr v1ptr) 1 (castDevPtr v3ptr) 1
        return v3

axpy :: (Vect,Vect) -> Vect
axpy (v1,v2) = foreignAcc cudaAxpy pureAxpy $ lift (v1,v2)
  where cudaAxpy = CUDAForeignAcc "cudaAxpyF" (\stream -> cudaAxpyF (Just stream))
        pureAxpy :: Acc (Vector Float, Vector Float) -> Acc (Vector Float)
        pureAxpy vs = let (u,v) = unlift vs
                      in zipWith (+) u v

test = do
  let x = fromList (Z:.5) [2,4,6,8,10] :: Vector Float
  let y = fromList (Z:.5) [1..5]       :: Vector Float

  run $ axpy ((use x), (use y))