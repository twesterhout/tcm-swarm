{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ViewPatterns #-}

-- |
-- Module      : PSO.Neural
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.Neural
  ( mutableMcmcLoop
  , energyHH1DOpen
  , RBM(..)
  , mkRbm
  , uniformRbm
  , MCMC(..)
  , unsafeFreezeMcmc
  , unsafeFreezeRbm
  , Rbm
  , Mcmc
  , HH1DOpen(..)
  , listAll
  ) where

import Foreign.C.Types
import Foreign.Storable

import Control.Monad
import Control.Monad.Primitive
import Control.Monad.Reader
import Control.Monad.ST
import Data.Bits
import Data.Coerce
import Data.Complex
import qualified Data.Vector.Storable as V
import Pipes
import qualified Pipes.Prelude as P
import qualified System.Random.MWC as MWC

import System.IO.Unsafe

import PSO.Internal.Neural
import PSO.Random
import PSO.Swarm


type family RealOf x

type instance RealOf Float = Float

type instance RealOf Double = Double

type instance RealOf CFloat = CFloat

type instance RealOf CDouble = CDouble

type instance RealOf (Complex Float) = Float

type instance RealOf (Complex Double) = Double

type instance RealOf (Complex CFloat) = CFloat

type instance RealOf (Complex CDouble) = CDouble

type family RbmPtr a

type instance RbmPtr (Complex Float) = RbmC

data Rbm a =
  R !(RbmPtr a)

data MRbm s a =
  MR !(RbmPtr a)

class Storable a => RBM a where
  newRbm :: PrimMonad m => Int -> Int -> m (MRbm (PrimState m) a)
  cloneRbm :: PrimMonad m => Rbm a -> m (MRbm (PrimState m) a)
  setWeights :: PrimMonad m => MRbm (PrimState m) a -> V.Vector a -> m ()
  setVisible :: PrimMonad m => MRbm (PrimState m) a -> V.Vector a -> m ()
  setHidden :: PrimMonad m => MRbm (PrimState m) a -> V.Vector a -> m ()
  sizeVisible :: Rbm a -> Int
  sizeHidden :: Rbm a -> Int
  sizeWeights :: Rbm a -> Int
  sizeWeights rbm = sizeVisible rbm * sizeHidden rbm
  debugPrintRbm :: Rbm a -> IO ()

unsafeFreezeRbm :: PrimMonad m => MRbm (PrimState m) a -> m (Rbm a)
unsafeFreezeRbm (MR p) = return (R p)

mkRbm :: (PrimMonad m, RBM a) => V.Vector a -> V.Vector a -> V.Vector a -> m (Rbm a)
mkRbm a@(V.length -> n) b@(V.length -> m) w
  | V.length w == n * m = newRbm n m >>= \rbm ->
      setVisible rbm a >> setHidden rbm b >> setWeights rbm w >> unsafeFreezeRbm rbm
  | otherwise           = error "mkRbm: Dimensions mismatch."

uniformRbm ::
     (PrimMonad m, RBM a, UniformDist m a) => Int -> Int -> (a, a) -> m (Rbm a)
uniformRbm n m bounds = do
  a <- uniformVector n bounds
  b <- uniformVector m bounds
  w <- uniformVector (n * m) bounds
  mkRbm a b w

class RBM a => AXPBY λ a where
  axpby :: PrimMonad m => λ -> Rbm a -> λ -> MRbm (PrimState m) a -> m ()

class RBM a => SCAL λ a where
  scal :: PrimMonad m => λ -> MRbm (PrimState m) a -> m ()

type family McmcPtr a

type instance McmcPtr (Complex Float) = McmcC

data Mcmc a =
  M !(McmcPtr a) !(RbmPtr a)

data MMcmc s a =
  MM !(McmcPtr a) !(RbmPtr a)

class RBM a => MCMC a where
  newMcmc :: PrimMonad m => Rbm a -> V.Vector a -> m (MMcmc (PrimState m) a)
  logWF :: Mcmc a -> a
  logQuotient1 :: Mcmc a -> Int -> a
  logQuotient2 :: Mcmc a -> Int -> Int -> a
  propose1 :: Mcmc a -> Int -> RealOf a
  propose2 :: Mcmc a -> Int -> Int -> RealOf a
  update1 :: PrimMonad m => MMcmc (PrimState m) a -> Int -> m ()
  update2 :: PrimMonad m => MMcmc (PrimState m) a -> Int -> Int -> m ()
  debugPrintMcmc :: Mcmc a -> IO ()
  -- getRbm :: Mcmc a -> Rbm a

unsafeFreezeMcmc :: PrimMonad m => MMcmc (PrimState m) a -> m (Mcmc a)
unsafeFreezeMcmc (MM p q) = return (M p q)

instance RBM (Complex Float) where
  newRbm nrVis nrHid = unsafeIOToPrim $ liftM MR (_newRbmC nrVis nrHid)
  cloneRbm (R p) = unsafeIOToPrim $ liftM MR (_cloneRbmC p)
  setWeights (MR p) w = unsafeIOToPrim $ _setWeightsC p (coerce w)
  setVisible (MR p) a = unsafeIOToPrim $ _setVisibleC p (coerce a)
  setHidden (MR p) b = unsafeIOToPrim $ _setHiddenC p (coerce b)
  sizeVisible (R p) = _sizeVisibleC p
  sizeHidden (R p) = _sizeHiddenC p
  debugPrintRbm (R p) = _printRbmC p

instance AXPBY (Complex Float) (Complex Float) where
  axpby a (R x) b (MR y) = unsafeIOToPrim $ _caxpbyRbmC (coerce a) x (coerce b) y

instance SCAL (Complex Float) (Complex Float) where
  scal a (MR x) = unsafeIOToPrim $ _cscaleRbmC (coerce a) x

instance SCAL Float (Complex Float) where
  scal a (MR x) = unsafeIOToPrim $ _cscaleRbmC (coerce a :+ 0) x

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
  propose2 (M p _) flip1 flip2 = _proposeC2 p flip1 flip2
  update1 (MM p _) flip1 = unsafeIOToPrim $ _acceptC1 p flip1
  update2 (MM p _) flip1 flip2 = unsafeIOToPrim $ _acceptC2 p flip1 flip2
  debugPrintMcmc (M p _) = _printMcmcC p

instance (PrimMonad m, Randomisable m Float)
  => DeltaWell m Float (Rbm (Complex Float)) where
    upDeltaWell κ p@(R pPtr) x@(R xPtr)
      | sizeVisible p == sizeVisible x
          && sizeHidden p == sizeHidden x =
            let n = sizeVisible p
                m = sizeHidden p
                size = 2 * (n * m + n + m)
             in V.replicateM size random >>= \ v ->
                 unsafeIOToPrim (_upDeltaWellC κ pPtr xPtr v) >> return p

-- mcmc ::
--      (Monad m)
--   => (α -> m β)
--   -> (α -> β -> m Bool)
--   -> (α -> β -> m α)
--   -> (α -> γ)
--   -> α -> Producer γ m ()
-- mcmc propose accept update finalise ψ = do
--   f <- lift $ propose ψ
--   p <- lift $ accept ψ f
--   if p then yield (finalise ψ) >> lift (update ψ f) >>= \ψ' ->
--               mcmc propose accept update finalise ψ'
--        else mcmc propose accept update finalise ψ
-- 
-- mcmcH ::
--      ( PrimMonad m
--      , UniformDist m Int
--      , Randomisable m α
--      , Randomisable m Spin
--      , Storable α
--      , RealFloat α
--      , LA.Numeric (Complex α)
--      , Num (V.Vector (Complex α))
--      )
--   => Int
--   -> RBM (Complex α)
--   -> m (Complex α)
-- mcmcH steps machine = do
--   let dim = V.length (machine ^. visBias)
--       states = mcmc
--         (const $ uniform (0, dim - 1))
--         (wfAcceptor machine)
--         (wfUpdater machine)
--         fst
--       elem s = locEnergyHH1DOpen machine s
--   pure V.fromList <*> replicateM dim random >>= return . V.map spin2Num >>= \σ ->
--     P.sum (states (σ, mkTheta machine σ) >-> P.take steps >-> P.map elem)
--   --  pure fst <*> (P.fold' (\(a,b) (c,d) -> (a+c, b+d)) (0,0) id $
--   --    states σ >-> P.take steps >-> P.map (\x -> (elem x, prob x :+ 0)))
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

-- (x -> a -> x) -> x -> (x -> b)
-- 
-- (\(x, y) (a :+ b) -> (x + toRational a, y + toRational b))
-- (toRational 0, toRational 0)
-- (\(x, y) -> fromRational x :+ fromRational y)

debugAdd :: forall λ. (Real λ, Fractional λ, Show λ)
         => (Rational, Rational) -> (Complex λ) -> (Rational, Rational)
debugAdd (x, y) (a :+ b) = unsafePerformIO $ do
  putStrLn $ show (fromRational x :: λ) ++ "\t" ++ show (fromRational y :: λ)
  return (x + toRational a, y + toRational b)


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



