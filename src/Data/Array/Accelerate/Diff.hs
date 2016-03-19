{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns        #-}
{-# LANGUAGE TypeFamilies        #-}

module Data.Array.Accelerate.Diff where

import Prelude hiding (map, zipWith)
import Data.Array.Accelerate hiding ((++),(!!),collect,reverse)
import qualified Data.Array.Accelerate.AST as AST
import Data.Array.Accelerate.Smart as Smart
import Data.Array.Accelerate.Trafo as Trafo
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.Array.Sugar as Sugar hiding (Atuple)
import Data.Array.Accelerate.Product as Product

import Data.Array.Accelerate.BLAS.Internal.Types
import Data.Array.Accelerate.BLAS.Internal.Dot (sdot)
import Data.Array.Accelerate.BLAS.Internal.Gemm (gemv, gevm, gevv)
import Data.Array.Accelerate.BLAS.Internal.Axpy (axpy)

import Debug.Trace
import Data.Coerce
import Data.Dynamic
import qualified Data.Vector as V

testfn :: (Vect, Vect) -> Scal
testfn (u,v) = sdot ((map tanh u), v)

mlp :: Vect -> (Matr, Vect, Matr, Vect) -> Vect
mlp x (w1,b1,w2,b2) = layer w2 b2 $ layer w1 b1 x where
	layer w b h = map tanh $ axpy (b, (gemv (w,h)))

data Param
   = Weight     Matr
   | Bias       Vect

--liftAfun :: (Plain a -> Acc b) -> (Acc c -> Acc b)
liftAfun :: Unlift Acc a => (a -> b) -> (Acc (Plain a) -> b)
liftAfun f = f . unlift

coerceAfun :: (Acc a -> Acc b) -> (PreAcc Acc Seq Exp a -> PreAcc Acc Seq Exp b)
coerceAfun f = coerce f

toPreAcc :: Acc a -> PreAcc Acc Seq Exp a
toPreAcc = coerce

toAcc :: PreAcc Acc Seq Exp a -> Acc a
toAcc = coerce

toPreExp :: Exp a -> PreExp Acc Seq Exp a
toPreExp = coerce

toExp :: PreExp Acc Seq Exp a -> Exp a
toExp = coerce

prepareAfun :: (Unlift Acc a, Arrays (Plain a)) => (a -> Acc b) -> Acc b
prepareAfun f = liftAfun f $ Acc $ Atag 0

prepareEfun :: (Elt a) => (Exp a -> Exp b) -> Exp b
prepareEfun f = f $ Exp $ Tag 0

--diff :: (a -> Acc b) -> Acc b -> String
diff :: (Unlift Acc a, Arrays (Plain a))
  => (a -> Vect)
  -> Vect
  -> [Param]
  -> IO Vect
diff f dout params = diffAcc (prepareAfun f) dout params

--maybe make this an IO and have it emit Acc expressions for each tag it encounters
--but continue to have the return type of the overall Acc
diffAcc :: forall arrs.
  Vect
  -> Vect
  -> [Param]
  -> IO Vect 
diffAcc self dout params = case toPreAcc self of
  -- PreAcc Acc Seq Exp (Plain (Matr, Vect, Matr, Vect))
  Atag i                          -> trace "Atag" $ undefined -- Atag i -- AST.Avar (prjIdx ("de Bruijn conversion tag " ++ show i) i alyt)
  Pipe afun1 afun2 acc            -> trace "Pipe" $ undefined -- Pipe afun1 afun2 acc 

  --Aforeign ff afun acc            -> trace "diffAcc:Aforeign" $ (res,dparams) where -- Aforeign ff afun acc where
  --  (res,dparams) = case strForeign ff of
  --    "axpy" -> trace "Aforeign:axpy" $ 

  Aforeign ff@(strForeign -> "axpy") afun
    (toPreAcc -> Atuple (SnocAtup (SnocAtup NilAtup arg1') arg2')) -> trace "diffAcc:axpy" $ outval where
      outval = do
        res1' <- res1
        res2' <- res2
        return $ axpy (res1', res2')
      arg1 = case (fromDynamic $ toDyn arg1') :: Maybe Vect of
        Nothing -> trace "diffAcc:axpy!type mismatch" $ undefined
        Just fn -> fn
      arg2 = case (fromDynamic $ toDyn arg2') :: Maybe Vect of
        Nothing -> trace "diffAcc:axpy!type mismatch" $ undefined
        Just fn -> fn
      res1 = diffAcc arg1 dout params
      res2 = diffAcc arg2 dout params

  Aforeign ff@(strForeign -> "gemv") afun
    (toPreAcc -> Atuple (SnocAtup (SnocAtup NilAtup argm') argv')) -> do
      traceM "diffAcc:gemv"
      let argm = case (fromDynamic $ toDyn argm') :: Maybe Matr of
                  Nothing -> trace "diffAcc:gemv!type mismatch" $ undefined
                  Just fn -> fn
      let argv = case (fromDynamic $ toDyn argv') :: Maybe Vect of
                  Nothing -> trace "diffAcc:gemv!type mismatch" $ undefined
                  Just fn -> fn
      let ix = case toPreAcc argm of
                  Aprj (toInt -> ix') (toPreAcc -> Atag 0) -> ix'
      let resm = case params !! ix of
                  Weight w -> w
                  Bias b -> trace "diffAcc:gemv!bias in weight slot" $ undefined
      resv <- diffAcc argv (gevm (dout, resm)) params
      traceM $ "diffAcc:gemv:Aprj: gradient for parameter " ++ show ix ++ " is ("
      traceShowM $ gevv (dout, resv)
      traceM $ ")"
      return $ gemv (resm, resv)

  --Acond b acc1 acc2               -> trace "Acond" $ Acond b acc1 acc2       -- AST.Acond (cvtE b) (cvtA acc1) (cvtA acc2)
  --Awhile pred iter init           -> trace "Awhile" $ Awhile pred iter init      -- AST.Awhile (cvtAfun1 pred) (cvtAfun1 iter) (cvtA init)
  Atuple arrs                     -> trace "Atuple" $ undefined --Atuple arrs      -- AST.Atuple (convertSharingAtuple config alyt aenv' arrs)
  Aprj (toInt -> ix) (toPreAcc -> Atag 0) -> do
    traceM $ "diffAcc:Aprj: gradient for parameter " ++ show ix ++ " is ("
    traceShowM dout
    traceM $ ")"
    return $ case params !! ix of
      Weight w -> trace "diffAcc:Aprj!weight in bias slot" $ undefined
      Bias b -> b

  Use array                       -> do
    traceM "Use"
    return $ use array -- Use array         -- AST.Use (fromArr array)
  --Unit e                          -> trace "Unit" $ undefined -- Unit e        -- AST.Unit (cvtE e)
  Generate sh f                   -> trace "Generate" $ undefined -- Generate sh f    -- AST.Generate (cvtE sh) (cvtF1 f)
  --Reshape e acc                   -> trace "Reshape" $ Reshape e acc      -- AST.Reshape (cvtE e) (cvtA acc)
  --Replicate ix acc                -> trace "Replicate" $ Replicate ix acc   -- mkReplicate (cvtE ix) (cvtA acc)
  --Slice acc ix                    -> trace "Slice" $ Slice acc ix        -- mkIndex (cvtA acc) (cvtE ix)

  Map fn'@(prepareEfun -> f) acc -> trace "diffAcc:Map" $ outval where
    fn = case (fromDynamic $ toDyn fn') :: Maybe (Exp Float -> Exp Float) of
      Nothing -> trace "diffAcc:Map!type mismatch" $ undefined
      Just fn -> fn
    outval = do
      inval' <- inval
      return . toAcc $ Map fn inval' --acc
    --(inval,dparams) = diffAcc acc (toAcc $ Map (diffExp f) dout) params
    inval = case (fromDynamic $ toDyn acc) :: Maybe Vect of 
      Nothing -> trace "diffAcc:Map!size mismatch" $ undefined
      Just acc' -> diffAcc acc' (toAcc $ Map (diffExp f) dout) params

  --  f' = prepareEfun f
  --  df = case f' of
  --    PrimApp p e -> case p of
  --                    AST.PrimTanh _ -> \x -> 1 - (tanh x) ** 2
  --                    otherwise -> undefined
  --  df = case f of
  --    DiffExpFn f' -> diffExpFn f'
  --    BadExpFn f' -> undefine
       -- Map f acc          -- AST.Map (cvtF1 f) (cvtA acc)
  -- f :: Exp e -> Exp e'
  --ZipWith f acc1 acc2             -> trace "ZipWith" $ ZipWith f acc1 acc2     -- AST.ZipWith (cvtF2 f) (cvtA acc1) (cvtA acc2)
  --Fold f e acc                    -> trace "Fold" $ Fold f e acc         -- AST.Fold (cvtF2 f) (cvtE e) (cvtA acc)
  --Fold1 f acc                     -> trace "Fold1" $ Fold1 f acc        -- AST.Fold1 (cvtF2 f) (cvtA acc)
  --FoldSeg f e acc1 acc2           -> trace "FoldSeg" $ FoldSeg f e acc1 acc2     -- AST.FoldSeg (cvtF2 f) (cvtE e) (cvtA acc1) (cvtA acc2)
  --Fold1Seg f acc1 acc2            -> trace "Fold1Seg" $ Fold1Seg f acc1 acc2    -- AST.Fold1Seg (cvtF2 f) (cvtA acc1) (cvtA acc2)
  --Scanl f e acc                   -> trace "Scanl" $ Scanl f e acc       -- AST.Scanl (cvtF2 f) (cvtE e) (cvtA acc)
--  Scanl' f e acc                  -> trace "Scanl'" $ Scanl' f e acc      -- AST.Scanl' (cvtF2 f) (cvtE e) (cvtA acc)
  --Scanl1 f acc                    -> trace "Scanl1" $ Scanl1 f acc      -- AST.Scanl1 (cvtF2 f) (cvtA acc)
  --Scanr f e acc                   -> trace "Scanr" $ Scanr f e acc       -- AST.Scanr (cvtF2 f) (cvtE e) (cvtA acc)
--  Scanr' f e acc                  -> trace "Scanr'" $ Scanr' f e acc      -- AST.Scanr' (cvtF2 f) (cvtE e) (cvtA acc)
  --Scanr1 f acc                    -> trace "Scanr1" $ Scanr1 f acc      -- AST.Scanr1 (cvtF2 f) (cvtA acc)
  Permute f dftAcc perm acc       -> trace "Permute" $ undefined --Permute f dftAcc perm acc     -- AST.Permute (cvtF2 f) (cvtA dftAcc) (cvtF1 perm) (cvtA acc)
  Backpermute newDim perm acc     -> trace "Backpermute" $ undefined --Backpermute newDim perm acc -- AST.Backpermute (cvtE newDim) (cvtF1 perm) (cvtA acc)
  Stencil stencil boundary acc    -> trace "Stencil" $ undefined -- Stencil stencil boundary acc
  Stencil2 stencil bndy1 acc1 bndy2 acc2
    -> trace "Stencil2" $ undefined -- Stencil2 stencil bndy1 acc1 bndy2 acc2
  Collect seq                     -> trace "Collect" $ undefined -- Collect seq      -- AST.Collect (convertSharingSeq config alyt EmptyLayout aenv' [] seq)
  otherwise -> trace "diffAcc:otherwise" undefined

--combine :: (Acc ta, arrs) -> (Acc ta, arrs) -> (Acc ta, arrs)
--combine (a, dp1) (b, dp2) = (val, merge dp1 dp2) where
--  val = toAcc $ Atuple (SnocAtup (SnocAtup NilAtup a) b)
--  merge NilAtup NilAtup = NilAtup
--  merge (SnocAtup rest1 snoc1) (SnocAtup rest2 snoc2) = 
--    (SnocAtup (merge rest1 rest2) $ zipWith (+) snoc1 snoc2) -- add cond for zero

--combine :: Vect -> Vect -> TupleRepr (Vect, Vect)-- (((), Array DIM1 Float), Array DIM1 Float)
--combine a b = toAcc $ Atuple (SnocAtup (SnocAtup NilAtup a) b)

--dyncast :: (Typeable ta, Typeable tb) => forall ta tb. ta -> tb -> ta
--dyncast a b = case (fromDynamic $ toDyn b) `asTypeOf` Just a of
--  Nothing -> trace "dyncast!type mismatch" $ undefined
--  Just c -> c

diffExp :: Exp Float -> (Exp Float -> Exp Float)
diffExp f = case toPreExp f of
  PrimApp p e -> trace "diffExp:PrimApp" $ diffPrim p
  otherwise -> trace "diffExp:otherwise" $ undefined

diffPrim :: AST.PrimFun (e -> Float) -> (Exp Float -> Exp Float)
diffPrim f = case f of
  AST.PrimTanh _ -> trace "diffPrim:tanh' = sech^2" $ \x -> 1 - (tanh x) ** 2

--collect :: Atuple a b -> [b]
--collect NilAtup          = []
--collect (SnocAtup tup b) = collect tup ++ [b]

toInt :: TupleIdx t e -> Int
toInt ZeroTupIdx       = 0
toInt (SuccTupIdx tup) = toInt tup + 1

test :: IO ()
test = do
  let x = fromList (Z:.10) [1..10] :: Vector Float
  let y = fromList (Z:.10) [2..11] :: Vector Float
  let dz = fromList (Z:.10) $ repeat 1 :: Vector Float

  let w = fromList (Z:.3:.3) [0,5,1,2,6,-1,-4,3,7] :: Array DIM2 Float
  let v = fromList (Z:.3:.2) [8,-2,4,5,2,-3] :: Array DIM2 Float
  --traceShowM $ sdot (map tanh $ use x) (use y)
  --traceShowM $ I.run1 (liftAfun testfn) (x,y)

  traceShowM $ liftAfun $ mlp $ use x
  r <- diff (mlp $ use x) (use dz)
    $ reverse [Weight $ use w,Bias $ use x,Weight $ use v,Bias $ use y]
  -- w :: Acc (Aprj (SuccTupIdx ...) _)
  --   :: Acc (Aprj (SuccTupIdx ...) (Acc (Atag ...)))
  -- h :: Acc (Aprj ZeroTupIdx _)

  let r = toPreAcc $ prepareAfun $ mlp $ use x
  () <- case r of
    Map f acc -> do
      --traceShowM acc
      () <- case toPreExp $ prepareEfun f of
        Tag i -> traceM "Tag"
        Const c -> traceM "Const"
        Tuple tup -> traceM "Tuple"
        Prj i e -> traceM "Prj"
        --IndexNil -> traceM "IndexNil"
        PrimConst c -> traceM "PrimConst"
        PrimApp p e -> do
          traceM "PrimApp"
          () <- case p of
            AST.PrimTanh _ -> traceM "PrimTanh"
          let ex = toPreExp e
          case ex of
            Tag i -> traceShowM i
            otherwise -> traceM "otherwise2"
        Index a e -> traceM "Index"
        otherwise -> traceM "otherwise"
      case toPreAcc acc of 
        Aforeign (strForeign -> "axpy") afun (toPreAcc -> Atuple (SnocAtup (SnocAtup NilAtup arg1) arg2)) -> do
          --let c = collect $ arrs
          --let tup = tuple . collect arrs
          --let blah = arrs :: Int
          --traceShowM $ arg1
          --traceShowM $ arg2
          () <- case toPreAcc arg1 of
            Atag i -> traceShowM i
            Aforeign ff afun arrs -> traceM $ strForeign ff
            Aprj (toInt -> ix) (toPreAcc -> Atag i) -> do
              traceShowM (i,ix) -- i: var index, ix: tuple index
          case toPreAcc arg2 of
            Atag i -> traceShowM i
            Aforeign ff afun arrs -> traceM $ strForeign ff
            Aprj ix a -> traceM "Aprj"
          --case toPreAcc acc2 of
          --  Map f acc -> traceM "Map"
          --  Aforeign ff2 afun2 acc3 -> traceM $ strForeign ff2
          --  Atuple arrs -> traceM "Atuple"
          --  otherwise -> traceM "otherwise4"
        otherwise -> traceM "otherwise3"



  --traceShowM r
  -- traceShowM $ diff (coerce $ axpy $ use (x,y)) (use dz)

  --() <- case r of
  --	AST.Alam f -> do
  --		traceShowM "Alam"
  --		case f of
  --			AST.Abody b -> do
  --				traceShowM "Abody"
  --				traceShowM $ diff b (Val ((),(x,y))) dz
  --				--case b of
  --				--	Trafo.Manifest pacc -> traceShowM "Manifest"

  --let Trafo.Manifest pacc = f

  --let f = evalOpenAfun r Empty

  return ()

  --let Trafo.Manifest pacc = r

  --traceShowM pacc

  --let f = convertAfunWith config afun
  --  in  evalOpenAfun f Empty

  --let ex = sdot (use x) (use y)
  --let r = convertAccWith config ex
  --let Trafo.Manifest pacc = r
  --case pacc of
  --	Alet acc1 acc2 -> traceShowM (acc1, acc2)

