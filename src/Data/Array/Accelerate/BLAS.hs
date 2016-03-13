module Data.Array.Accelerate.BLAS where

import Foreign (Ptr)
import Foreign.CUDA

newtype Handle = Handle { useHandle :: ((Ptr ()))}

foreign import ccall "cublas_v2.h cublasCreate_v2" cublasCreate :: IO Handle

foreign import ccall "cublas_v2.h cublasSdot_v2"
   cublasSdot :: Handle 
      -> Int -> DevicePtr Float
      -> Int -> DevicePtr Float
      -> Int -> DevicePtr Float -> IO ()
