{-# OPTIONS_GHC -threaded #-}
import Control.Monad (when)    
import System.Exit (exitFailure)
import Test.QuickCheck (quickCheckResult)
import Test.QuickCheck.Test (isSuccess)
import Data.Array.Accelerate hiding (all, not, length)
import Prelude hiding (zipWith)
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.CUDA as C
import Data.Array.Accelerate.BLAS.Internal.Dot
import Data.Array.Accelerate.BLAS.Internal.Gemm

prop_dot_shape :: [Float] -> [Float] -> Bool
prop_dot_shape v1 v2 = length (toList $ I.run $ sdot (use (fromList (Z:.(length v1)) v1)) (use (fromList (Z:.(length v2)) v2))) == 1

prop_dot_440 :: Bool
prop_dot_440 = head (toList $ I.run $ sdot (use (fromList (Z:.10) [1..])) (use (fromList (Z:.10) [2..11]))) == 440.0

prop_gemm :: Bool
prop_gemm = (toList $ I.run $ gemm (use (fromList (Z:.3:.3) [0,5,1,2,6,-1,-4,3,7] :: Array DIM2 Float)) (use (fromList (Z:.3:.2) [8,-2,4,5,2,-3] :: Array DIM2 Float))) == [22.0,22.0,38.0,29.0,-6.0,2.0]

main :: IO ()
main = do
  let tests = [ quickCheckResult prop_dot_shape,
                quickCheckResult prop_dot_440,
                quickCheckResult prop_gemm
              ]
  success <- fmap (all isSuccess) . sequence $ tests
  when (not success) $ exitFailure