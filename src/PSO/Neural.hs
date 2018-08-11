{-# LANGUAGE GADTs #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE UnboxedTuples #-}

-- |
-- Module      : PSO.Neural
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.Neural
  (
  -- mutableMcmcLoop
  -- , energyHH1DOpen
  -- , energyHH1DOpenC
  -- , energyHH1DOpenMKLC
  -- , energyHH1DOpenZ
  -- , energyVarHH1DOpen
  -- , RBM(..)
    Rbm
  , mkRbm
  , uniformRbm
  , IsRbm(..)
  , fromRbm
  , toRbm
  , unsafeFreezeRbm
  , unsafeThawRbm
  , mcmcHeisenberg1D
  -- , MCMC(..)
  -- , unsafeFreezeMcmc
  -- , Mcmc
  -- , HH1DOpen(..)
  -- , listAll
  , Measurement(..)
  , HasMean(..)
  , HasVar(..)
  ) where

import Prelude hiding (map, mapM, zipWith, zipWithM)
import qualified Prelude as Prelude

import Control.Exception(assert)
import Control.Monad (Monad(..), (>=>), (<=<))
import Control.Monad.Primitive
import Control.Monad.ST

import Data.Bits
import Data.Coerce
import Data.Complex
import Data.Functor.Identity
import Data.NumInstances.Tuple
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import Data.Vector.Fusion.Stream.Monadic (Stream(..), Step(..), SPEC(..))
import qualified Data.Vector.Fusion.Stream.Monadic as Stream
import qualified Data.Vector.Fusion.Bundle.Monadic as Bundle

import Foreign.C.Types
import Foreign.Storable
import Foreign.ForeignPtr

import GHC.Generics (Generic)

import Lens.Micro
import Lens.Micro.TH

import System.IO.Unsafe

import PSO.Internal.Neural
import PSO.Random
import PSO.Swarm


type family RealOf x

type instance RealOf Float = Float

type instance RealOf Double = Double

type instance RealOf CFloat = CFloat

type instance RealOf CDouble = CDouble

type instance RealOf (Complex a) = a

type family RbmCore a

type instance RbmCore Float = RbmC

-- type instance RbmCore (Complex Double) = RbmZ

newtype Rbm a = R (ForeignPtr (RbmCore a)) deriving (Generic)

newtype MRbm s a = MR (ForeignPtr (RbmCore a)) deriving (Generic)


class Storable a => IsRbm a where
  newRbm :: PrimMonad m => Int -> Int -> m (MRbm (PrimState m) a)
  cloneRbm :: PrimMonad m => Rbm a -> m (MRbm (PrimState m) a)
  getWeights :: PrimMonad m => MRbm (PrimState m) a -> m (V.MVector (PrimState m) a, V.MVector (PrimState m) a)
  getVisible :: PrimMonad m => MRbm (PrimState m) a -> m (V.MVector (PrimState m) a, V.MVector (PrimState m) a)
  getHidden  :: PrimMonad m => MRbm (PrimState m) a -> m (V.MVector (PrimState m) a, V.MVector (PrimState m) a)
  sizeWeights :: Rbm a -> Int
  sizeVisible :: Rbm a -> Int
  sizeHidden  :: Rbm a -> Int
  -- debugPrintRbm :: Rbm a -> IO ()

unsafeThawRbm :: PrimMonad m => Rbm a -> m (MRbm (PrimState m) a)
unsafeThawRbm (R p) = return (MR p)

unsafeFreezeRbm :: PrimMonad m => MRbm (PrimState m) a -> m (Rbm a)
unsafeFreezeRbm (MR p) = return (R p)

-- setWeights :: PrimMonad m => MRbm (PrimState m) a -> V.Vector a -> V.Vector a -> m ()
-- setWeights mrbm wR wI = do
--   (mwR, mwI) <- getWeights mrbm
--   V.copy mwR wR
--   V.copy mwI wI
-- 
-- setVisible :: PrimMonad m => MRbm (PrimState m) a -> V.Vector a -> V.Vector a -> m ()
-- setVisible mrbm aR aI = do
--   (maR, maI) <- getVisible mrbm
--   V.copy maR aR
--   V.copy maI aI
-- 
-- setHidden :: PrimMonad m => MRbm (PrimState m) a -> V.Vector a -> V.Vector a -> m ()
-- setHidden mrbm bR bI = do
--   (mbR, mbI) <- getHidden mrbm
--   V.copy mbR bR
--   V.copy mbI bI


fromRbm :: (Monad m, IsRbm a) => Rbm a -> Stream.Stream m a
fromRbm rbm = Bundle.elements $ Bundle.fromVectors (runST $ unsafeThawRbm rbm >>= go)
  where
    go :: IsRbm a => MRbm s a -> ST s [V.Vector a]
    go mrbm = do
      (wR, wI) <- getWeights mrbm
      (aR, aI) <- getVisible mrbm
      (bR, bI) <- getHidden mrbm
      Prelude.mapM V.freeze [wR, wI, aR, aI, bR, bI]

toRbm :: (PrimMonad m, IsRbm a) => Int -> Int -> Stream.Stream m a -> m (Rbm a)
toRbm n m (Stream step t) = do
  mrbm <- newRbm n m
  (wR, wI) <- getWeights mrbm
  (aR, aI) <- getVisible mrbm
  (bR, bI) <- getHidden mrbm
  copy SPEC wR (n * m) 0 t >>=
    copy SPEC wI (n * m) 0 >>=
      copy SPEC aR n 0 >>=
        copy SPEC aI n 0 >>=
          copy SPEC bR m 0 >>=
            copy SPEC bI m 0
  unsafeFreezeRbm mrbm
  where
    copy !_ !v !size !i s
      | i < size = step s >>= \r ->
        case r of
          Yield x s' -> MV.write v i x >> copy SPEC v size (i + 1) s'
          Skip    s' -> copy SPEC v size i s'
          Done       -> error "No, we're not done yet!"
      | otherwise = return s

-- | Creates a new Restricted Boltzmann Machine with given weights.
mkRbm :: IsRbm a
      => V.Vector (Complex a) -- ^ Visible bias \(a^N\).
      -> V.Vector (Complex a) -- ^ Hidden bias \(a^M\).
      -> V.Vector (Complex a) -- ^ Weights \(a^{M\times N}\). *Column-major layout is assumed!*.
      -> Rbm a
mkRbm a@(V.length -> nx1) b@(V.length -> mx1) w@(V.length -> mxn)
  | mxn == nx1 * mx1 = runST $ toRbm nx1 mx1 $ Bundle.elements $
    Bundle.fromVectors [ realPart `V.map` w, imagPart `V.map` w
                       , realPart `V.map` a, imagPart `V.map` a
                       , realPart `V.map` b, imagPart `V.map` b
                       ]
  | otherwise = error "mkRbm: Dimensions mismatch."

-- | Creates an RBM with weights distributed uniformly within given intervals.
uniformRbm ::
     (PrimMonad m, IsRbm a, Ord a, UniformDist m (Complex a))
  => Int -- ^ Number of visible nodes.
  -> Int -- ^ Number of hidden nodes.
  -> (a, a) -- ^ Range for visible bias.
  -> (a, a) -- ^ Range for hidden bias.
  -> (a, a) -- ^ Range for weights.
  -> m (Rbm a)
uniformRbm !n !m !(amin, amax) !(bmin, bmax) !(wmin, wmax)
  | n < 0 || m < 0 = error "uniformRbm: Negative dimensions??"
  | amin > amax || bmin > bmax || wmin > wmax =
    error "uniformRbm: Lower bound is per definition not \
          \greater than the upper bound."
  | otherwise = do
    a <- uniformVector n (amin :+ amin, amax :+ amin)
    b <- uniformVector m (bmin :+ bmin, bmin :+ bmax)
    w <- uniformVector (n * m) (wmin :+ wmin, wmax :+ wmax)
    return $ mkRbm a b w

instance IsRbm Float where
  {-# INLINE newRbm #-}
  newRbm n m = unsafeIOToPrim $ MR <$> _RbmC'construct n m
  {-# INLINE cloneRbm #-}
  cloneRbm (R p) = unsafeIOToPrim $ MR <$> _RbmC'clone p
  {-# INLINE getWeights #-}
  getWeights (MR p) = _RbmC'getWeights p
  {-# INLINE getVisible #-}
  getVisible (MR p) = _RbmC'getVisible p
  {-# INLINE getHidden #-}
  getHidden (MR p) = _RbmC'getHidden p
  {-# INLINE sizeVisible #-}
  sizeVisible (R fp) = unsafePerformIO $
    withForeignPtr fp (peek >=> return . _rbmC'sizeVisible)
  {-# INLINE sizeHidden #-}
  sizeHidden (R fp) = unsafePerformIO $
    withForeignPtr fp (peek >=> return . _rbmC'sizeHidden)
  {-# INLINE sizeWeights #-}
  sizeWeights x = sizeHidden x * sizeVisible x

getDims :: IsRbm a => Rbm a -> (Int, Int)
getDims x = (sizeVisible x, sizeHidden x)
{-# INLINE getDims #-}

zipWithM :: (IsRbm a, IsRbm b, IsRbm c, PrimMonad m)
         => (a -> b -> m c) -> Rbm a -> Rbm b -> m (Rbm c)
zipWithM f x y = assert (getDims x == getDims y) $
  toRbm (sizeVisible x) (sizeHidden x) $ Stream.zipWithM f (fromRbm x) (fromRbm y)

zipWith :: forall a b c. (IsRbm a, IsRbm b, IsRbm c)
        => (a -> b -> c) -> Rbm a -> Rbm b -> Rbm c
zipWith f x y = runST $ zipWithM f' x y
  where f' :: a -> b -> ST s c
        f' a b = return $ f a b

mapM :: (IsRbm a, IsRbm b, PrimMonad m)
     => (a -> m b) -> Rbm a -> m (Rbm b)
mapM f x = toRbm (sizeVisible x) (sizeHidden x) $ Stream.mapM f $ fromRbm x

map :: (IsRbm a, IsRbm b)
     => (a -> b) -> Rbm a -> Rbm b
map f x = runST $ mapM (return . f) x

instance Num (Rbm Float) where
  (+) x y = zipWith (+) x y
  (-) x y = zipWith (-) x y
  (*) x y = zipWith (*) x y
  abs x = map abs x
  signum x = map signum x
  negate x = map negate x
  fromInteger = undefined

instance Scalable Float (Rbm Float) where
  scale λ = map (*λ)

instance (PrimMonad m, DeltaWell m Float Float)
  => DeltaWell m Float (Rbm Float) where
    upDeltaWell κ p x = zipWithM (upDeltaWell κ) p x

{-
instance RBM (Complex Double) where
  newRbm nrVis nrHid = unsafeIOToPrim $ liftM MR (_newRbmZ nrVis nrHid)
  cloneRbm (R p) = unsafeIOToPrim $ liftM MR (_cloneRbmZ p)
  setWeights (MR p) w = unsafeIOToPrim $ _setWeightsZ p (coerce w)
  setVisible (MR p) a = unsafeIOToPrim $ _setVisibleZ p (coerce a)
  setHidden (MR p) b = unsafeIOToPrim $ _setHiddenZ p (coerce b)
  sizeVisible (R p) = _sizeVisibleZ p
  sizeHidden (R p) = _sizeHiddenZ p
  debugPrintRbm (R p) = _printRbmZ p

instance AXPBY (Complex Float) (Complex Float) where
  axpby a (R x) b (MR y) = unsafeIOToPrim $ _caxpbyRbmC (coerce a) x (coerce b) y

instance AXPBY (Complex Double) (Complex Double) where
  axpby a (R x) b (MR y) = unsafeIOToPrim $ _zaxpbyRbmZ (coerce a) x (coerce b) y

instance SCAL (Complex Float) (Complex Float) where
  scal a (MR x) = unsafeIOToPrim $ _cscaleRbmC (coerce a) x

instance SCAL Float (Complex Float) where
  scal a (MR x) = unsafeIOToPrim $ _cscaleRbmC (coerce a :+ 0) x

instance SCAL (Complex Double) (Complex Double) where
  scal a (MR x) = unsafeIOToPrim $ _zscaleRbmZ (coerce a) x

instance SCAL Double (Complex Double) where
  scal a (MR x) = unsafeIOToPrim $ _zscaleRbmZ (coerce a :+ 0) x

instance (Num a, AXPBY a a, SCAL a a) => Num (Rbm a) where
  (+) x y = runST $ cloneRbm x >>= \z ->
    axpby (1 :: a) y (1 :: a) z >> unsafeFreezeRbm z
  (-) x y = runST $ cloneRbm x >>= \z ->
    axpby ((-1) :: a) y (1 :: a) z >> unsafeFreezeRbm z
  (*) x y = undefined
  negate x = runST $ cloneRbm x >>= \z -> scal ((-1) :: a) z >> unsafeFreezeRbm z
  fromInteger = undefined
  abs = undefined
  signum = undefined

instance (SCAL λ a) => Scalable λ (Rbm a) where
  scale k x = runST $ cloneRbm x >>= \z -> scal k z >> unsafeFreezeRbm z

instance MCMC (Complex Float) where
  newMcmc (R rbm) spin = unsafeIOToPrim $
    _newMcmcC rbm (coerce spin) >>= \mcmc -> return (MM mcmc rbm)
  logWF (M p _) = coerce $ _logWFC p
  logQuotient1 (M p _) flip1 = coerce $ _logQuotientWF1C p flip1
  logQuotient2 (M p _) flip1 flip2 = coerce $ _logQuotientWF2C p flip1 flip2
  propose1 m@(M p _) = (^2) . magnitude . exp . logQuotient1 m -- _proposeC1 p flip1
  propose2 (M p _) flip1 flip2 = _propose2C p flip1 flip2
  update1 (MM p _) flip1 = unsafeIOToPrim $ _accept1C p flip1
  update2 (MM p _) flip1 flip2 = unsafeIOToPrim $ _accept2C p flip1 flip2
  debugPrintMcmc (M p _) = _printMcmcC p

instance MCMC (Complex Double) where
  newMcmc (R rbm) spin = unsafeIOToPrim $
    _newMcmcZ rbm (coerce spin) >>= \mcmc -> return (MM mcmc rbm)
  logWF (M p _) = coerce $ _logWFZ p
  logQuotient1 (M p _) flip1 = coerce $ _logQuotientWF1Z p flip1
  logQuotient2 (M p _) flip1 flip2 = coerce $ _logQuotientWF2Z p flip1 flip2
  propose1 m@(M p _) = (^2) . magnitude . exp . logQuotient1 m -- _proposeZ1 p flip1
  propose2 (M p _) flip1 flip2 = _propose2Z p flip1 flip2
  update1 (MM p _) flip1 = unsafeIOToPrim $ _accept1Z p flip1
  update2 (MM p _) flip1 flip2 = unsafeIOToPrim $ _accept2Z p flip1 flip2
  debugPrintMcmc (M p _) = _printMcmcZ p


mutableMcmcLoop ::
     (PrimMonad m)
  => (α -> Int -> m β)
  -> (α -> β -> m Bool)
  -> (α -> β -> m ())
  -> α
  -> Int
  -> Producer α m ()
mutableMcmcLoop propose accept update ψ n = do
  proposal <- lift $ propose ψ n
  isAcceptable <- lift $ accept ψ proposal
  yield ψ
  when isAcceptable (lift (update ψ proposal))
  mutableMcmcLoop propose accept update ψ (n + 1)

class MCMC a => HH1DOpen a where
  locEnergyHH1DOpen :: Mcmc a -> a

instance HH1DOpen (Complex Float) where
  locEnergyHH1DOpen (M p _) = coerce (_locEHH1DOpenC p)

instance HH1DOpen (Complex Double) where
  locEnergyHH1DOpen (M p _) = coerce (_locEHH1DOpenZ p)

unpack2Bools :: (Num α, Bits α) => Int -> α -> V.Vector Bool
unpack2Bools n x = V.fromList $ doUnpack n x []
  where doUnpack 0 _ []   = [False]
        doUnpack 0 _ bits = bits
        doUnpack n x bits = doUnpack (n - 1) (x `shiftR` 1) $
                              (testBit x 0) : bits

listAll :: forall m a.
     ( PrimMonad m
     , HH1DOpen a
     , Num a
     )
  => Rbm a -> m (V.Vector a)
listAll rbm =
  let n = sizeVisible rbm
      toSpin True  = 1
      toSpin False = -1
      σs :: [V.Vector a]
      σs = V.map toSpin <$> unpack2Bools n <$> [(0 :: Int) .. (2 ^ n - 1)]
   in V.fromList <$> (newMcmc rbm >=> unsafeFreezeMcmc >=> return . logWF) `mapM` σs

energyHH1DOpen ::
     forall m a λ.
     ( PrimMonad m
     , Ord (RealOf a)
     , Fractional a
     , HH1DOpen a
     , Randomisable m Bool
     , Randomisable m (RealOf a)
     , UniformDist m Int
     -- , MonadIO m
     -- , a ~ Complex λ
     -- , Fractional λ
     -- , Real λ
     -- , Show λ
     )
  => Rbm a -> Int -> Int -> m a
energyHH1DOpen rbm offset steps =
  let n = sizeVisible rbm
      proposer :: MMcmc (PrimState m) a -> Int -> m Int
      proposer _ i = {- return (i `mod` n) -} uniform (0, n - 1)
      acceptor :: MMcmc (PrimState m) a -> Int -> m Bool
      acceptor mx flip = unsafeFreezeMcmc mx >>= \x ->
                            ((<= propose1 x flip) <$> random)
      updater :: MMcmc (PrimState m) a -> Int -> m ()
      updater = update1
      func :: MMcmc (PrimState m) a -> m a
      func = unsafeFreezeMcmc >=> return . locEnergyHH1DOpen
      mmcmc :: m (MMcmc (PrimState m) a)
      mmcmc = randomSpin n >>= newMcmc rbm
      states = mutableMcmcLoop proposer acceptor updater
   in mmcmc >>= \mmcmc' ->
                  (/ fromIntegral steps)
                    <$> P.sum
                  --     P.fold debugAdd
                  --            (toRational 0, toRational 0)
                  --            (\(x, y) -> fromRational (x / toRational steps)
                  --                         :+ fromRational (y / toRational steps))
                             (states mmcmc' 0 >-> P.drop offset
                                              >-> P.mapM func
                                              >-> P.take steps)

energyHH1DOpenC :: forall m.
     ( PrimMonad m
     , Randomisable m Bool
     , Randomisable m Float
     , UniformDist m Int
     )
  => Rbm (Complex Float) -> Int -> Int -> m (Complex Float)
energyHH1DOpenC rbm offset steps =
  let n = sizeVisible rbm
      mmcmc = randomSpin n >>= newMcmc rbm
      ints :: m (V.Vector Int)
      ints = uniformVector (offset + steps) (0, n - 1)
      floats :: m (V.Vector Float)
      floats = V.replicateM (offset + steps) random
   in ints >>= \ is ->
        floats >>= \ fs ->
          mmcmc >>= \ (MM p (RbmC m)) ->
            unsafeIOToPrim $ (_runMcmcC p offset steps is fs) >>= \ x ->
              touchForeignPtr m >> return (coerce x)

energyHH1DOpenZ :: forall m.
     ( PrimMonad m
     , Randomisable m Bool
     , Randomisable m Double
     , UniformDist m Int
     )
  => Rbm (Complex Double) -> Int -> Int -> m (Complex Double)
energyHH1DOpenZ rbm offset steps =
  let n = sizeVisible rbm
      mmcmc = randomSpin n >>= newMcmc rbm
      ints :: m (V.Vector Int)
      ints = uniformVector (offset + steps) (0, n - 1)
      floats :: m (V.Vector Double)
      floats = V.replicateM (offset + steps) random
   in ints >>= \ is ->
        floats >>= \ fs ->
          mmcmc >>= \ (MM p (RbmZ m)) ->
            unsafeIOToPrim $ (_runMcmcZ p offset steps is fs) >>= \ x ->
              touchForeignPtr m >> return (coerce x)
-}

mcmcHeisenberg1D ::
     PrimMonad m => Rbm Float -> Int -> Int -> m (Measurement Float)
mcmcHeisenberg1D (R rbm) offset steps = unsafeIOToPrim $ _RbmC'heisenberg1D rbm offset steps

instance Eq a => Eq (Measurement a) where
  (==) (Measurement _ x) (Measurement _ y) = x == y

instance Ord a => Ord (Measurement a) where
  -- (<=) (Measurement a x) (Measurement b y) = x <= y
  (<=) (Measurement a x) (Measurement b y) = a <= b

instance Num a => Num (Measurement a) where
  (+) (Measurement a b) (Measurement c d) = Measurement (a + c) (b + d)
  (-) (Measurement a b) (Measurement c d) = Measurement (a - c) (b - d)
  (*) (Measurement a b) (Measurement c d) = Measurement (a * c) (b * d)
  negate (Measurement a b) = Measurement (negate a) (negate b)
  fromInteger n = Measurement (fromInteger n) (fromInteger n)
  abs (Measurement a b) = Measurement (abs a) (abs b)
  signum (Measurement a b) = Measurement (signum a) (signum b)

instance Fractional a => Fractional (Measurement a) where
  (/) (Measurement a b) (Measurement c d) = Measurement (a / c) (b / d)
  recip (Measurement a b) = Measurement (recip a) (recip b)

makeFields ''Measurement

{-

debugAdd :: forall λ. (Real λ, Fractional λ, Show λ)
         => (λ, λ) -> (λ, λ) -> (λ, λ)
debugAdd (x, y) (a, b) = unsafePerformIO $ do
  putStrLn $ show x ++ "\t" ++ show y ++ "\t" ++ show a ++ "\t" ++ show b
  return (x + a, y + b)

energyVarHH1DOpen ::
     forall m a.
     ( PrimMonad m
     , Ord a
     , RealFloat a
     , HH1DOpen (Complex a)
     , Randomisable m Bool
     , Randomisable m a
     , UniformDist m Int
     -- , MonadIO m
     , Show a
     )
  => Rbm (Complex a) -> Int -> Int -> m (Measurement a)
energyVarHH1DOpen rbm offset steps =
  let n = sizeVisible rbm
      proposer :: MMcmc (PrimState m) (Complex a) -> Int -> m Int
      proposer _ i = return (i `mod` n) {- uniform (0, n - 1) -}
      acceptor :: MMcmc (PrimState m) (Complex a) -> Int -> m Bool
      acceptor mx flip = unsafeFreezeMcmc mx >>= \x ->
                            ((<= propose1 x flip) <$> random)
      updater :: MMcmc (PrimState m) (Complex a) -> Int -> m ()
      updater = update1
      func :: MMcmc (PrimState m) (Complex a) -> m (a, a)
      func = unsafeFreezeMcmc >=> return . locEnergyHH1DOpen
                              >=> return . (\ x -> (realPart x, (realPart x)^2))
      mmcmc :: m (MMcmc (PrimState m) (Complex a))
      mmcmc = randomSpin n >>= newMcmc rbm
      states = mutableMcmcLoop proposer acceptor updater
   in mmcmc >>= \mmcmc' ->
                  P.fold debugAdd
                         0
                         (\(h, h2) -> let mean = (h  / fromIntegral steps)
                                          var  = (h2 / fromIntegral steps) - mean^2
                                       in MV mean var)
                         (states mmcmc' 0 >-> P.drop offset
                                          >-> P.mapM func
                                          >-> P.take steps)
-}
