{-# LANGUAGE TypeOperators #-}
module Data.Array.Accelerate.BLAS.Internal.Gemm where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith,fold,all,replicate)
import Foreign.CUDA.Ptr
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
-- import Debug.Trace

type Matrix a = Array DIM2 a

matMul :: (IsNum e, Elt e) => Acc (Matrix e, Matrix e) -> Acc (Matrix e)
matMul vs = fold (+) 0 $ zipWith (*) arrRepl brrRepl
  where
    (arr,brr)           = unlift vs
    Z :. rowsA :. _     = unlift (shape arr)    :: Z :. Exp Int :. Exp Int
    Z :. _     :. colsB = unlift (shape brr)    :: Z :. Exp Int :. Exp Int

    arrRepl             = replicate (lift $ Z:. All   :. colsB :. All) arr
    brrRepl             = replicate (lift $ Z:. rowsA :. All   :. All) (transpose brr)

cudaGemmF :: Maybe Stream -> (Matrix Float, Matrix Float) -> CIO (Matrix Float)
cudaGemmF ms (a,b) = do
  -- traceShowM "cudaGemmF"
  let Z :. ra :. ca = arrayShape a   -- m k
      Z :. rb :. cb = arrayShape b   -- k n
  -- traceShowM (ra,ca,rb,cb)
  c <- allocateArray $ Z :. ra :. cb -- m n
  withDevicePtrs a ms $ \aptr -> do
    withDevicePtrs b ms $ \bptr -> do
      withDevicePtrs c ms $ \cptr -> do
        -- BL.gemm :: Handle -> transa -> transb -> m -> n -> k -> alpha -> a -> lda -> b -> ldb -> beta -> c -> ldc
        -- Since CUBLAS uses column-major mode on matrices, we will need to change this up a bit to:
        -- BL.gemm :: Handle -> transa -> transb -> n -> m -> k -> alpha -> b -> ldb -> a -> lda -> beta -> c -> ldc
        -- where
        -- transa = BL.N
        -- transb = BL.N
        -- m      = numRowsofA
        -- n      = numColsofB
        -- k      = numColsofA = numRowsofB
        -- lda    = numColsofA
        -- ldb    = numColsofB
        -- ldc    = numColsofB
        liftIO $ BL.gemm theHandle BL.N BL.N cb ra rb (CFloat 1) (castDevPtr bptr) cb (castDevPtr aptr) ca (CFloat 0) (castDevPtr cptr) cb
        return c

gemm :: Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float)
gemm v1 v2 = foreignAcc cudaGemm matMul $ lift (v1,v2)
 where cudaGemm = CUDAForeignAcc "cudaGemmF" (\stream -> cudaGemmF (Just stream))

test = do
 -- answer should be [22,22,38,29,-6,2]
 let x = fromList (Z:.3:.3) [0,5,1,2,6,-1,-4,3,7] :: Array DIM2 Float
 let y = fromList (Z:.3:.2) [8,-2,4,5,2,-3] :: Array DIM2 Float

 run $ gemm (use x) (use y)
