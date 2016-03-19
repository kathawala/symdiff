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
import Data.Array.Accelerate hiding ((++),collect)
import qualified Data.Array.Accelerate.AST as AST
import Data.Array.Accelerate.Smart as Smart
import Data.Array.Accelerate.Trafo as Trafo
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.Array.Sugar as Sugar hiding (Atuple)
import Data.Array.Accelerate.Product as Product

import Data.Array.Accelerate.BLAS.Internal.Dot (sdot)
import Data.Array.Accelerate.BLAS.Internal.Gemm (gemm, Matr, Vect, Scal)
import Data.Array.Accelerate.BLAS.Internal.Axpy (axpy)

import Debug.Trace
import Data.Coerce
import Data.Dynamic

tupdot :: (Vect, Vect) -> Scal
tupdot (u,v) = sdot u v

testfn :: (Vect, Vect) -> Scal
testfn (u,v) = sdot (map tanh u) v

mlp :: Vect -> (Matr, Vect, Matr, Vect) -> Vect
mlp x (w1,b1,w2,b2) = layer w2 b2 $ layer w1 b1 x where
	layer w b h = map tanh $ axpy b (gemm w h)

--mlp :: Acc (Vector Float, Vector Float) -> Acc (Vector Float)
--mlp vs = axpy b1 b2 where
--	(b1,b2) = unlift vs


--evalOpenAfun :: DelayedOpenAfun aenv f -> Val aenv -> f
--evalOpenAfun (AST.Alam  f) aenv = \a -> evalOpenAfun f (aenv `Push` a)
--evalOpenAfun (AST.Abody b) aenv = evalOpenAcc b aenv

--evalAtuple :: Atuple (DelayedOpenAcc aenv) t -> [String]
--evalAtuple NilAtup             = []
--evalAtuple (SnocAtup tup a)    = evalAtuple tup ++ [evalOpenAcc a]

--works:
--evalAtuple :: Atuple (DelayedOpenAcc aenv) t -> Val aenv -> t
--evalAtuple NilAtup        _    = ()
--evalAtuple (SnocAtup t a) aenv = (evalAtuple t aenv, evalOpenAcc a aenv)

--myevalTupIdx :: Product.TupleIdx t e -> Int
--myevalTupIdx Product.ZeroTupIdx        = 0
--myevalTupIdx (Product.SuccTupIdx idx) = myevalTupIdx idx + 1

--myevalIdx :: AST.Idx t e -> Int
--myevalIdx AST.ZeroIdx        = 0
--myevalIdx (AST.SuccIdx idx) = myevalIdx idx + 1

--diff :: DelayedOpenAfun aenv f -> Val aenv -> f
--diff (AST.Alam  f) aenv = \a -> evalOpenAfun f (aenv `Push` a)
--diff (AST.Abody b) aenv = evalOpenAcc b aenv

--diff'
--    :: forall aenv a b.
--    (Arrays a, Arrays b) =>
--    DelayedOpenAcc aenv a
--    -> Val aenv
--    -> b
--    -> Acc (b,b)
--diff' (Trafo.Manifest pacc) aenv dout = -- "got a manifest array"
--  case pacc of
----    Avar ix                     -> "Avar " ++ show (myevalIdx ix) -- prj ix aenv
----    Alet acc1 acc2              -> diff acc2 (aenv `Push` diff acc1 aenv)
----    AST.Atuple atup                 -> "Atuple of " ++ show (evalAtuple atup aenv) -- toAtuple $ evalAtuple atup aenv
----    AST.Aprj ix atup                -> "Aprj index " ++ show (myevalTupIdx ix) ++ " " ++ diff atup aenv -- myevalPrj ix (fromAtuple $ diff atup)
----    Apply afun acc              -> evalOpenAfun afun aenv  $ manifest acc
--    AST.Aforeign ff afun acc         -> case strForeign ff of
--    	"axpy" -> use (dout,dout) -- evalOpenAfun afun Empty $ manifest acc
----    Acond p acc1 acc2
----      | evalE p                 -> manifest acc1
----      | otherwise               -> manifest acc2

----    Awhile cond body acc        -> go (manifest acc)
----      where
----        p       = evalOpenAfun cond aenv
----        f       = evalOpenAfun body aenv
----        go !x
----          | p x ! Z     = go (f x)
----          | otherwise   = x

----    Use arr                     -> toArr arr
----    Unit e                      -> unitOp (evalE e)
----    Collect s                   -> evalSeq defaultSeqConfig s aenv

--    ---- Producers
--    ---- ---------
--    --AST.Map f acc                   -> "Map of a fn over" ++ diff acc aenv -- (evalF f) (delayed acc)
--    --Generate sh f               -> generateOp (evalE sh) (evalF f)
--    --Transform sh p f acc        -> transformOp (evalE sh) (evalF p) (evalF f) (delayed acc)
--    --Backpermute sh p acc        -> backpermuteOp (evalE sh) (evalF p) (delayed acc)
--    --Reshape sh acc              -> reshapeOp (evalE sh) (manifest acc)

--    --ZipWith f acc1 acc2         -> zipWithOp (evalF f) (delayed acc1) (delayed acc2)
--    --Replicate slice slix acc    -> replicateOp slice (evalE slix) (manifest acc)
--    --Slice slice acc slix        -> sliceOp slice (manifest acc) (evalE slix)

--    ---- Consumers
--    ---- ---------
--    --Fold f z acc                -> foldOp (evalF f) (evalE z) (delayed acc)
--    --Fold1 f acc                 -> fold1Op (evalF f) (delayed acc)
--    --FoldSeg f z acc seg         -> foldSegOp (evalF f) (evalE z) (delayed acc) (delayed seg)
--    --Fold1Seg f acc seg          -> fold1SegOp (evalF f) (delayed acc) (delayed seg)
--    --Scanl f z acc               -> scanlOp (evalF f) (evalE z) (delayed acc)
--    --Scanl' f z acc              -> scanl'Op (evalF f) (evalE z) (delayed acc)
--    --Scanl1 f acc                -> scanl1Op (evalF f) (delayed acc)
--    --Scanr f z acc               -> scanrOp (evalF f) (evalE z) (delayed acc)
--    --Scanr' f z acc              -> scanr'Op (evalF f) (evalE z) (delayed acc)
--    --Scanr1 f acc                -> scanr1Op (evalF f) (delayed acc)
--    --Permute f def p acc         -> permuteOp (evalF f) (manifest def) (evalF p) (delayed acc)
--    --Stencil sten b acc          -> stencilOp (evalF sten) b (manifest acc)
--    --Stencil2 sten b1 acc1 b2 acc2-> stencil2Op (evalF sten) b1 (manifest acc1) b2 (manifest acc2)

--GOOD STUFF:
--diff :: ( Elt tf, IsFloating tf, Arrays tr)
--            => PreAcc Acc Seq Exp tf
--            -> Acc tf
--            -> Acc tr

---- diff' tk tag@(Tag f level) _  

---- diff (Const _) _ = constant 0

--diff (Aforeign a b c) (Acc dout) = case strForeign a of
--	"axpy" -> use (dout,dout)
--END GOOD STUFF

-- diff' tk (Tuple t) _ = constant 11 -- tup d0 d1 where

-- Not sure we can know at compile time that t is compatible with x
-- Assume tup0 is a 2-tuple with identical types
-- TODO
--diff' tk (Prj (i::TupleIdx (TupleRepr tup0) tf) (te::Exp tup0)) x = Exp (Prj i dodgyTupleDiff) where
--  dodgyTupleDiff = unsafeCoerce (diffTT tk teDodgy x) :: Exp tup0 where
--  teDodgy = unsafeCoerce te :: Exp (tf, tf)

--diff' tk (Foreign a b c) (Exp x) = case x of
--  Foreign xa _ _ -> case matchMarkers a xa of
--    True -> constant 1
--    False -> diffT tk (b c) $ Exp x

--class Diff f where
--  type DiffR f
--  aconvert :: Smart.Layout aenv aenv -> f -> AST.OpenAfun aenv (DiffR f)

--instance (Arrays a, Diff r) => Diff (Acc a -> r) where
--  type DiffR (Acc a -> r) = a -> DiffR r
--  --
--  aconvert config alyt f
--    = let a     = Acc $ Atag (sizeLayout alyt)
--          alyt' = incLayout alyt `PushLayout` ZeroIdx
--      in
--      Alam $ aconvert config alyt' (f a)

--instance Arrays b => Diff (Acc b) where
--  type DiffR (Acc b) = b
--  --
--  aconvert config alyt body
--    = let lvl    = sizeLayout alyt
--          vars   = [lvl-1, lvl-2 .. 0]
--      in
--      Abody $ convertOpenAcc config lvl vars alyt body

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
  => (a -> Acc (Array sh Float))
  -> Acc (Array sh Float)
  -> a
  -> IO (Acc (Array sh Float)) 
diff f dout params = diffAcc (prepareAfun f) dout params

--maybe make this an IO and have it emit Acc expressions for each tag it encounters
--but continue to have the return type of the overall Acc
diffAcc :: forall sh arrs.
  Acc (Array sh Float)
  -> Acc (Array sh Float)
  -> arrs
  -> IO (Acc (Array sh Float)) 
diffAcc self dout params = case toPreAcc self of
  -- PreAcc Acc Seq Exp (Plain (Matr, Vect, Matr, Vect))
  --Atag i                          -> trace "Atag" $ Atag i -- AST.Avar (prjIdx ("de Bruijn conversion tag " ++ show i) i alyt)
  --Pipe afun1 afun2 acc            -> trace "Pipe" $ Pipe afun1 afun2 acc 

  --Aforeign ff afun acc            -> trace "diffAcc:Aforeign" $ (res,dparams) where -- Aforeign ff afun acc where
  --  (res,dparams) = case strForeign ff of
  --    "axpy" -> trace "Aforeign:axpy" $ 

  Aforeign ff@(strForeign -> "axpy") afun
    (toPreAcc -> Atuple (SnocAtup (SnocAtup NilAtup arg1') arg2')) -> trace "diffAcc:axpy" $ outval where
      outval = do
        inval' <- inval
        return . toAcc $ Aforeign ff afun' inval'
      afun = case (fromDynamic $ toDyn afun') :: Maybe (Acc (Vector Float, Vector Float) -> Acc (Vector Float))
      inval = do
        res1' <- res1
        res2' <- res2
        return . toAcc $ Atuple (SnocAtup (SnocAtup NilAtup res1') res2')
      arg1 = case (fromDynamic $ toDyn arg1') :: Maybe (Acc (Array sh Float)) of
        Nothing -> trace "diffAcc!axpy type mismatch" $ undefined
        Just fn -> fn
      arg2 = case (fromDynamic $ toDyn arg2') :: Maybe (Acc (Array sh Float)) of
        Nothing -> trace "diffAcc!axpy type mismatch" $ undefined
        Just fn -> fn
      res1 = diffAcc arg1 dout params
      res2 = diffAcc arg2 dout params

  --Acond b acc1 acc2               -> trace "Acond" $ Acond b acc1 acc2       -- AST.Acond (cvtE b) (cvtA acc1) (cvtA acc2)
  --Awhile pred iter init           -> trace "Awhile" $ Awhile pred iter init      -- AST.Awhile (cvtAfun1 pred) (cvtAfun1 iter) (cvtA init)
  --Atuple arrs                     -> trace "Atuple" $ Atuple arrs      -- AST.Atuple (convertSharingAtuple config alyt aenv' arrs)
  --Aprj ix a                       -> trace "Aprj" $ Aprj ix a        -- AST.Aprj ix (cvtA a)
  --Use array                       -> trace "Use" $ Use array         -- AST.Use (fromArr array)
  --Unit e                          -> trace "Unit" $ Unit e        -- AST.Unit (cvtE e)
  --Generate sh f                   -> trace "Generate" $ Generate sh f    -- AST.Generate (cvtE sh) (cvtF1 f)
  --Reshape e acc                   -> trace "Reshape" $ Reshape e acc      -- AST.Reshape (cvtE e) (cvtA acc)
  --Replicate ix acc                -> trace "Replicate" $ Replicate ix acc   -- mkReplicate (cvtE ix) (cvtA acc)
  --Slice acc ix                    -> trace "Slice" $ Slice acc ix        -- mkIndex (cvtA acc) (cvtE ix)

  Map fn'@(prepareEfun -> f) acc -> trace "diffAcc:Map" $ outval where
    fn = case (fromDynamic $ toDyn fn') :: Maybe (Exp Float -> Exp Float) of
      Nothing -> trace "diffAcc:Map type mismatch" $ undefined
      Just fn -> fn
    outval = do
      inval' <- inval
      return . toAcc $ Map fn inval' --acc
    --(inval,dparams) = diffAcc acc (toAcc $ Map (diffExp f) dout) params
    inval = case (fromDynamic $ toDyn acc) :: Maybe (Acc (Array sh Float)) of 
      Nothing -> trace "diffAcc:Map size mismatch" $ undefined
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
  --Permute f dftAcc perm acc       -> trace "Permute" $ Permute f dftAcc perm acc     -- AST.Permute (cvtF2 f) (cvtA dftAcc) (cvtF1 perm) (cvtA acc)
  --Backpermute newDim perm acc     -> trace "Backpermute" $ Backpermute newDim perm acc -- AST.Backpermute (cvtE newDim) (cvtF1 perm) (cvtA acc)
  --Stencil stencil boundary acc    -> trace "Stencil" $ Stencil stencil boundary acc
  --Stencil2 stencil bndy1 acc1 bndy2 acc2
  --  -> trace "Stencil2" $ Stencil2 stencil bndy1 acc1 bndy2 acc2
  --Collect seq                     -> trace "Collect" $ Collect seq      -- AST.Collect (convertSharingSeq config alyt EmptyLayout aenv' [] seq)
  otherwise -> trace "diffAcc:otherwise" undefined

--data family TupMaybe tup :: *
--data instance TupMaybe (a,b)                 = TupMaybe2 (a,b)
--data instance TupMaybe (a,b,c)               = TupMaybe3 (a,b,c)--(Maybe a,Maybe b,Maybe c)
--data instance TupMaybe (a,b,c,d)             = TupMaybe4 (a,b,c,d)--(Maybe a,Maybe b,Maybe c,Maybe d)
--data instance TupMaybe (a,b,c,d,e)           = TupMaybe5 (a,b,c,d,e)--(Maybe a,Maybe b,Maybe c,Maybe d,Maybe e)
--data instance TupMaybe (a,b,c,d,e,f)         = TupMaybe6 (a,b,c,d,e,f)--(Maybe a,Maybe b,Maybe c,Maybe d,Maybe e,Maybe f)
--data instance TupMaybe (a,b,c,d,e,f,g)       = TupMaybe7 (a,b,c,d,e,f,g)--(Maybe a,Maybe b,Maybe c,Maybe d,Maybe e,Maybe f,Maybe g)
--data instance TupMaybe (a,b,c,d,e,f,g,h)     = TupMaybe8 (a,b,c,d,e,f,g,h)--(Maybe a,Maybe b,Maybe c,Maybe d,Maybe e,Maybe f,Maybe g,Maybe h)
--data instance TupMaybe (a,b,c,d,e,f,g,h,i)   = TupMaybe9 (a,b,c,d,e,f,g,h,i)--(Maybe a,Maybe b,Maybe c,Maybe d,Maybe e,Maybe f,Maybe g,Maybe h,Maybe i)
--data instance TupMaybe (a,b,c,d,e,f,g,h,i,j) = TupMaybe10 (a,b,c,d,e,f,g,h,i,j)--(Maybe a,Maybe b,Maybe c,Maybe d,Maybe e,Maybe f,Maybe g,Maybe h,Maybe i,Maybe j)

--type family Atup tup :: *
--type instance Atup (a,b)        = Atuple $ SnocAtup (SnocAtup NilAtup a) b
--type instance Atup (a,b,c)      = Atuple $ SnocAtup (SnocAtup (SnocAtup NilAtup a) b) c
--type instance Atup (a,b,c,d)    = Atuple $ SnocAtup (SnocAtup (SnocAtup (SnocAtup NilAtup a) b) c) d

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

--data ExpFn e e' where
--  DiffExpFn :: (Exp Float -> Exp Float) -> ExpFn Float Float
--  BadExpFn :: (Exp e -> Exp e') -> ExpFn e e'
--  diffExpFn :: f -> f
--instance DiffExpFn (\x -> )
--diffExpFn :: (Exp Float -> Exp Float) -> (Exp Float -> Exp Float)
--diffExpFn f = diffExp $ f $ Exp $ Tag 0

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

  --traceShowM $ liftAfun $ mlp $ use x
  --traceShowM $ diff (mlp $ use x) (use dz) (use w,use x,use v,use y)
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

