{-# LANGUAGE TypeOperators #-}
module Data.Array.Accelerate.BLAS.Internal.Gemm where

import Data.Array.Accelerate
import Data.Array.Accelerate.BLAS
import Data.Array.Accelerate.BLAS.Internal.Types
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate.CUDA.Foreign
import Prelude hiding (zipWith,fold,all,replicate, length, (++))
import Foreign.CUDA.Ptr
import qualified Foreign.CUDA.Cublas as BL
import qualified Foreign.CUDA.Cublas.FFI as BLF
import Foreign.C.Types
-- import Debug.Trace

pureGemv :: (IsNum e, Elt e) => Acc (Matrix e, Vector e) -> Acc (Vector e)
pureGemv vs = slice result (lift (Z :. All :. (0::Int)))
  where
    result              = (fold (+) 0 $ zipWith (*) arrRepl brrRepl)
    (arr,brr)           = unlift vs
    bLen                = length brr
    Z :. rowsA :. _     = unlift (shape arr)    :: Z :. Exp Int :. Exp Int

    arrRepl             = replicate (lift $ Z:. All   :. (1::Int) :. All) arr
    brrMat              = reshape   (lift $ Z:. bLen  :. (1::Int))        brr
    brrRepl             = replicate (lift $ Z:. rowsA :. All      :. All) (transpose brrMat)

cudaGemvF :: Maybe Stream -> (Matrix Float, Vector Float) -> CIO (Vector Float)
cudaGemvF ms (a,b) = do
  let Z :. ra :. ca = arrayShape a   -- m k
      Z :. rb       = arrayShape b   -- k n
  c <- allocateArray $ Z :. ra -- m n
  withDevicePtrs a ms $ \aptr -> do
    withDevicePtrs b ms $ \bptr -> do
      withDevicePtrs c ms $ \cptr -> do
        -- where
        -- transa = BL.N
        -- transb = BL.N
        -- m      = numRowsofA
        -- n      = numColsofB
        -- k      = numColsofA = numRowsofB
        -- lda    = numColsofA
        -- ldb    = numColsofB
        -- ldc    = numColsofB
        liftIO $ BL.gemm theHandle BL.N BL.N 1 ra rb (CFloat 1) (castDevPtr bptr) 1 (castDevPtr aptr) ca (CFloat 0) (castDevPtr cptr) 1
        return c

pureGevm :: (IsNum e, Elt e) => Acc (Vector e, Matrix e) -> Acc (Vector e)
pureGevm vs = slice result (lift (Z :. (0::Int) :. All))
  where
    result              = (fold (+) 0 $ zipWith (*) arrRepl brrRepl)
    (arr,brr)           = unlift vs
    aLen                = length arr
    Z :. _ :. colB      = unlift (shape brr)    :: Z :. Exp Int :. Exp Int

    arrMat              = reshape   (lift $ Z:. (1::Int)  :. aLen)            arr
    arrRepl             = replicate (lift $ Z:. All       :. colB     :. All) arrMat
    brrRepl             = replicate (lift $ Z:. (1::Int)  :. All      :. All) (transpose brr)

cudaGevmF :: Maybe Stream -> (Vector Float, Matrix Float) -> CIO (Vector Float)
cudaGevmF ms (a,b) = do
  let Z :. ra       = arrayShape a
      Z :. rb :. cb = arrayShape b

  c <- allocateArray $ Z :. ra -- m n
  withDevicePtrs a ms $ \aptr -> do
    withDevicePtrs b ms $ \bptr -> do
      withDevicePtrs c ms $ \cptr -> do
        liftIO $ BL.gemm theHandle BL.N BL.N cb ra 1 (CFloat 1) (castDevPtr bptr) cb (castDevPtr aptr) 1 (CFloat 0) (castDevPtr cptr) cb 
        return c

pureOuter :: (IsNum e, Elt e) => Acc (Vector e, Vector e) -> Acc (Matrix e) 
pureOuter vs = zipWith (*) arrRepl brrRepl
  where
    (arr,brr) = unlift vs
    aLen      = length arr
    bLen      = length brr

    arrRepl   = replicate (lift $ Z:. All  :. bLen) arr
    brrRepl   = replicate (lift $ Z:. aLen :.  All) brr

cudaOuterF :: Maybe Stream -> (Vector Float, Vector Float) -> CIO (Matrix Float)
cudaOuterF ms (a,b) = do
  let Z :. ra = arrayShape a
      Z :. cb = arrayShape b

  c <- allocateArray $ Z :. ra :. cb -- m n
  withDevicePtrs a ms $ \aptr -> do
    withDevicePtrs b ms $ \bptr -> do
      withDevicePtrs c ms $ \cptr -> do
        liftIO $ BL.gemm theHandle BL.T BL.N cb ra 1 (CFloat 1) (castDevPtr bptr) cb (castDevPtr aptr) 1 (CFloat 0) (castDevPtr cptr) cb 
        return c  

gemv :: (Matr, Vect) -> Vect
gemv (v1,v2) = foreignAcc cudaGemv pureGemv $ lift (v1,v2)
  where cudaGemv = CUDAForeignAcc "gemv" (\stream -> cudaGemvF (Just stream))

gevm :: (Vect, Matr) -> Vect
gevm (v1,v2) = foreignAcc cudaGevm pureGevm $ lift (v1,v2)
  where cudaGevm = CUDAForeignAcc "gevm" (\stream -> cudaGevmF (Just stream))

gevv :: (Vect, Vect) -> Matr
gevv (v1,v2) = foreignAcc cudaOuter pureOuter $ lift (v1,v2)
  where cudaOuter = CUDAForeignAcc "gevv" (\stream -> cudaOuterF (Just stream)) 

test = do
 -- answer should be [22,38,-6]
 let x = fromList (Z:.3:.3) [0,5,1,2,6,-1,-4,3,7] :: Array DIM2 Float
 let y = fromList (Z:.3) [8,-2,4]                 :: Array DIM1 Float

 run $ gemv (use x,use y)
