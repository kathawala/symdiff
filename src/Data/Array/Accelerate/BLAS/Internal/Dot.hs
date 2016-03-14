module Data.Array.Accelerate.BLAS.Internal.Dot where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA hiding (create)
import Data.Array.Accelerate hiding (snd)
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith)
--import System.Mem.Weak (addFinalizer)
-- import System.IO.Unsafe (unsafePerformIO)
import qualified Foreign.CUDA as C
import Foreign.CUDA.Ptr
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

--foreign import ccall "cublas_v2.h cublasSdot_v2"
--   cublasSdot :: BL.Handle 
--      -> Int -> DevicePtr Float
--      -> Int -> DevicePtr Float
--      -> Int -> DevicePtr Float -> IO ()

cudaDotProductF :: Maybe Stream -> (Vector Float, Vector Float) -> CIO (Scalar Float)
cudaDotProductF ms (v1,v2) = do
  traceShowM "cudaDotProductF"
  let n = arraySize (arrayShape v1)
  traceShowM n
  --o <- allocateArray Z
  -- need "handle" and pointer to v1 and "stride between consecutive elements of v1" and
  -- pointer to v2 and "stride between consecutive elements of v2" and result output array

  withDevicePtrs v1 ms $ \v1ptr -> do
    withDevicePtrs v2 ms $ \v2ptr -> do
      --o <- allocateArray Z
      --withDevicePtrs o ms $ \optr -> do
      case ms of
        Just s -> do
          liftIO $ BLF.setStream theHandle s
          CFloat o <- liftIO $ BL.dot theHandle n (castDevPtr v1ptr) 1 (castDevPtr v2ptr) 1
          return $ fromList Z [o]      
        Nothing -> do
          CFloat o <- liftIO $ BL.dot theHandle n (castDevPtr v1ptr) 1 (castDevPtr v2ptr) 1
          return $ fromList Z [o]
        --traceShowM "doing cublasSdot"
        --traceShowM "did cublasSdot"
        --return o
      --traceShowM "running BL.dot"
      --CFloat o <- liftIO $ BL.dot theHandle n (castDevPtr v1ptr) 1 (castDevPtr v2ptr) 1
      --traceShowM "ran BL.dot"
      --traceShowM o
      --return o
      --traceShowM $ fromList Z [o]
      --return $ fromList Z [o]

  --((),v1ptr) <- devicePtrsOfArray v1
  --((),v2ptr) <- devicePtrsOfArray v2
  ----((),optr)  <- devicePtrsOfArray o
  
  --traceShowM "running BL.dot"

  --CFloat o <- liftIO $ BL.dot theHandle n (castDevPtr v1ptr) 1 (castDevPtr v2ptr) 1

  --traceShowM "ran BL.dot"
  --traceShowM o
  --traceShowM $ fromList Z [o]
  --return $ fromList Z [o]

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
