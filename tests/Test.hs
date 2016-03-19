import Control.Monad (when)    
import System.Exit (exitFailure)
import Test.QuickCheck (quickCheckResult)
import Test.QuickCheck.Test (isSuccess)
import Data.Array.Accelerate hiding (all, not, length)
import Prelude hiding (zipWith)
-- import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.CUDA as I
import Data.Array.Accelerate.BLAS.Internal.Dot
import Data.Array.Accelerate.BLAS.Internal.Gemm
import Data.Array.Accelerate.BLAS.Internal.Axpy

prop_dot_shape :: [Float] -> [Float] -> Bool
prop_dot_shape v1 v2 = length (toList $ I.run $ sdot ((use (fromList (Z:.(length v1)) v1)), (use (fromList (Z:.(length v2)) v2)))) == 1

prop_dot_440 :: Bool
prop_dot_440 = head (toList $ I.run $ sdot ((use (fromList (Z:.10) [1..])), (use (fromList (Z:.10) [2..11])))) == 440.0

prop_gemv :: Bool
prop_gemv = (toList $ I.run $ gemv ((use (fromList (Z:.3:.3) [0,5,1,2,6,-1,-4,3,7] :: Array DIM2 Float)), (use (fromList (Z:.3) [8,-2,4] :: Vector Float)))) == [-6.0,0.0,-10.0]

prop_gevm :: Bool
prop_gevm = (toList $ I.run $ gevm ((use (fromList (Z:.3) [1..] :: Vector Float)), (use (fromList (Z:.3:.2) [2,1,0,-1,2,4] :: Array DIM2 Float)))) == [8.0,11.0]

prop_outer :: Bool
prop_outer = (toList $ I.run $ gevv ((use (fromList (Z:.4) [1..] :: Vector Float)), (use (fromList (Z:.3) [2,-2,1] :: Vector Float)))) == [2.0,-2.0,1.0,4.0,-4.0,2.0,6.0,-6.0,3.0,8.0,-8.0,4.0]

prop_axpy_integrity :: [Float] -> [Float] -> Bool
prop_axpy_integrity _ []  = True
prop_axpy_integrity [] _  = True
prop_axpy_integrity v1 v2 = (v2 /= v3 || all (\x -> x == 0) v1)
  where v3 = toList $ I.run $ axpy ((use (fromList (Z:.(length v1)) v1)), (use (fromList (Z:.(length v2)) v2)))

prop_axpy :: Bool
prop_axpy = (toList $ I.run $ axpy ((use (fromList (Z:.5) [2,4,6,8,10])), (use (fromList (Z:.5) [1..5])))) == [3.0,6.0,9.0,12.0,15.0]

main :: IO ()
main = do
  let tests = [ quickCheckResult prop_dot_shape,
                quickCheckResult prop_dot_440,
                quickCheckResult prop_gemv,
                quickCheckResult prop_gevm,
                quickCheckResult prop_outer,
                quickCheckResult prop_axpy_integrity,
                quickCheckResult prop_axpy
              ]
  success <- fmap (all isSuccess) . sequence $ tests
  when (not success) $ exitFailure