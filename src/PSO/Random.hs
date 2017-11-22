-- {-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE TypeSynonymInstances #-} -- needed fo GenIO
-- {-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-} -- MWC.Gen create
{-# LANGUAGE UndecidableInstances #-} -- for UniformDist
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE PolyKinds #-}

{-# OPTIONS_HADDOCK show-extensions #-}


-- |
-- Module      : PSO.Random
-- Description : Uniform interface for "Control.Random.Mersenne" and
--               "Control.Random.MWC"
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental

module PSO.Random
    ( Generator(..)
    , Randomisable(..)
    , UniformDist(..)
    , RandomScalable(..)
    , mkMTGen
    , mkMWCGen
    , mkMWCGenST
    ) where

import System.IO.Unsafe
import Data.Word(Word32)
import Data.Proxy
import Data.Complex(Complex(..))
import Control.Monad.Primitive
import Control.Monad.ST
import Control.Monad.Reader
import Data.Vector(singleton)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Generic as GV
import Foreign.C.Types(CDouble, CFloat)
import qualified Foreign
import qualified System.Random.Mersenne as Mersenne
import qualified System.Random.MWC as MWC


class (PrimMonad m) => Generator m g where
    create :: Maybe Word32 -> m g

instance Generator IO Mersenne.MTGen where
    create = Mersenne.newMTGen

instance Generator IO (MWC.Gen RealWorld) where
    create x = case x of
        (Just n) -> MWC.initialize . singleton $ n
        Nothing  -> MWC.create

instance Generator (ST s) (MWC.Gen s) where
    create x = case x of
        (Just n) -> MWC.initialize . singleton $ n
        Nothing  -> MWC.create


class (Monad m) => Randomisable m a where
    random :: m a


instance (Mersenne.MTRandom a)
  => Randomisable (ReaderT Mersenne.MTGen IO) a where
    random = ask >>= lift . Mersenne.random

instance (MWC.Variate a)
  => Randomisable (ReaderT (MWC.Gen RealWorld) IO) a where
    random = ask >>= lift . MWC.uniform

instance (MWC.Variate a)
  => Randomisable (ReaderT (MWC.Gen s) (ST s)) a where
    random = ask >>= lift . MWC.uniform

instance {-# OVERLAPS #-} (MWC.Variate a)
  => Randomisable (ReaderT (MWC.Gen RealWorld) IO) (Complex a) where
    random = liftM2 (:+) random random


mkMTGen :: Maybe Word32 -> IO Mersenne.MTGen
mkMTGen = create

mkMWCGen :: Maybe Word32 -> IO MWC.GenIO
mkMWCGen = create

mkMWCGenST :: Maybe Word32 -> ST s (MWC.GenST s)
mkMWCGenST = create


class (Monad m) => UniformDist m a where
  uniform :: (a, a) -> m a

class (Monad m) => RandomScalable m a where
  randScale :: a -> m a


uniformFloating :: (RealFloat a, Randomisable m a)
  => (a, a) -> m a
uniformFloating (low, high) = return (\x -> low + (high - low) * x) `ap` random


instance (Monad m, Randomisable m Float) => UniformDist m Float where
    uniform = uniformFloating

instance (Monad m, Randomisable m Double) => UniformDist m Double where
    uniform = uniformFloating

instance (Monad m, Randomisable m CFloat) => UniformDist m CFloat where
    uniform = uniformFloating

instance (Monad m, Randomisable m CDouble) => UniformDist m CDouble where
    uniform = uniformFloating

instance (RealFloat a, UniformDist m a)
  => UniformDist m (Complex a) where
    uniform ((rlow :+ ilow), (rhigh :+ ihigh)) =
      liftM2 (:+) (uniform (rlow, rhigh)) (uniform (ilow, ihigh))

instance (UniformDist m a) => UniformDist m [a] where
  uniform (low, high) = mapM uniform $ zip low high

instance (Foreign.Storable a, UniformDist m a)
  => UniformDist m (V.Vector a) where
    uniform (low, high) = V.zipWithM (\l h -> uniform (l, h)) low high


randScaleFloating :: (RealFloat a, Randomisable m a) => a -> m a
randScaleFloating x = liftM (* x) random

instance (Monad m, Randomisable m Float) => RandomScalable m Float where
    randScale = randScaleFloating

instance (Monad m, Randomisable m Double) => RandomScalable m Double where
    randScale = randScaleFloating

instance (Monad m, Randomisable m CFloat) => RandomScalable m CFloat where
    randScale = randScaleFloating

instance (Monad m, Randomisable m CDouble) => RandomScalable m CDouble where
    randScale = randScaleFloating

instance {-# OVERLAPS #-} (RandomScalable m a)
  => RandomScalable m (Complex a) where
    randScale (x :+ y) = return (:+) `ap` (randScale x) `ap` (randScale y)

instance (RandomScalable m a, GV.Vector v a) => RandomScalable m (v a) where
    randScale = GV.mapM randScale


