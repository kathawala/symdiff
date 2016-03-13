module Main where

import Foreign.CUDA.Cublas as C
import Foreign.CUDA.Driver as D

diff :: Int -> Int -> Int
diff a b = a - b

--main = do
--    D.initialise []
--    dev <- D.device 0
--    ctx <- D.create dev [] 
--    h <- C.create
--    return h
main = const () <$> C.create
