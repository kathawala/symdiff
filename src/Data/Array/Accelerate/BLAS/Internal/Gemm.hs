{-# LANGUAGE TypeOperators #-}
module Data.Array.Accelerate.BLAS.Internal.Gemm where

import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.CUDA hiding (create)
import Data.Array.Accelerate hiding (snd)
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith,fold,all,replicate)
import qualified Foreign.CUDA as C
import Foreign.CUDA.Ptr
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
import Debug.Trace

type Matrix a = Array DIM2 a

matMul :: (IsNum e, Elt e) => Acc (Matrix e, Matrix e) -> Acc (Matrix e)
matMul vs = fold (+) 0 $ zipWith (*) arrRepl brrRepl
  where
    (arr,brr) = unlift vs
    Z :. rowsA :. _     = unlift (shape arr)    :: Z :. Exp Int :. Exp Int
    Z :. _     :. colsB = unlift (shape brr)    :: Z :. Exp Int :. Exp Int

    arrRepl             = replicate (lift $ Z:. All   :. colsB :. All) arr
    brrRepl             = replicate (lift $ Z:. rowsA :. All   :. All) (transpose brr)

cudaGemmF :: Maybe Stream -> (Matrix Float, Matrix Float) -> CIO (Matrix Float)
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

gemm :: Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float)
gemm v1 v2 = foreignAcc cudaGemm matMul $ lift (v1,v2)
 where cudaGemm = CUDAForeignAcc "cudaGemmF" (\stream -> cudaGemmF (Just stream))

test = do
 -- answer should be [22,22,38,29,-6,2]
 let x = fromList (Z:.3:.3) [0,5,1,2,6,-1,-4,3,7] :: Array DIM2 Float
 let y = fromList (Z:.3:.2) [8,-2,4,5,2,-3] :: Array DIM2 Float

 run $ gemm (use x) (use y)
