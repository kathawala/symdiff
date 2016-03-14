-- {-# LANGUAGE ForeignFunctionInterface #-}

module Data.Array.Accelerate.BLAS where

--import Foreign
--import Foreign.Ptr
--import Foreign.C.Types
import Foreign.CUDA hiding (Status)
-- import Foreign.CUDA.Driver (initialise)
import System.IO.Unsafe (unsafePerformIO)
import Foreign.CUDA.Cublas as BL
import Foreign.CUDA.Cublas.FFI as BL
import Debug.Trace

--data HandleStruct = HandleStruct
--type Handle = Ptr HandleStruct

----newtype Handle = Handle (Ptr ()) -- { useHandle :: ((Ptr ()))}
--newtype Status = Status (CInt)

----instance Show Handle where
----   show (Handle ptr) = show ptr

--instance Show Status where
--    show (Status status) = show status

--foreign import ccall "cublas_v2.h cublasCreate_v2"
--    cublasCreate :: Ptr Handle -> IO Status

--create :: IO Handle
--create = do
--    _ <- initialise []
--    handleptr <- malloc
--    status <- cublasCreate handleptr
--    traceShowM status
--    handle <- peek handleptr
--    return handle

theHandle :: BL.Handle
theHandle = unsafePerformIO $ do
    h <- BL.create
    () <- BL.setPointerMode h BL.Device
    return h

--foreign import ccall "cublas.h cublasInit"
--    cublasInit :: IO ()

--foreign import ccall "cublas.h cublasSdot"
--   cublasSdot :: Int -> DevicePtr Float
--              -> Int -> DevicePtr Float
--              -> Int -> DevicePtr Float -> IO ()

    --alloca $ \handle -> do
    --    status <- cublasCreate handleptr
    --    traceShowM $ status
    --    handle <- peek handleptr
    --    return handle

--foreign import ccall "cublas_v2.h cublasSdot_v2"
--   cublasSdot :: Handle 
--      -> Int -> DevicePtr Float
--      -> Int -> DevicePtr Float
--      -> Int -> DevicePtr Float -> IO ()

--foreign import ccall "cublas_v2.h cublasSetPointerMode_v2"
--    cublasSetPointerMode :: Handle -> Ptr Int -> IO Status

--setPointerMode :: Handle -> Int -> IO ()
--setPointerMode handle pointerMode = alloca $ \pointerModePtr -> do
--    poke pointerModePtr pointerMode
--    status <- cublasSetPointerMode handle pointerModePtr
--    traceShowM status
--    return ()

--foreign import ccall "cublas_v2.h cublasGetPointerMode_v2"
--    cublasGetPointerMode :: Handle -> Ptr Int -> IO Status

--getPointerMode :: Handle -> IO Int
--getPointerMode handle = alloca $ \pointerModePtr -> do
--    status <- cublasGetPointerMode handle pointerModePtr
--    traceShowM status
--    peek pointerModePtr