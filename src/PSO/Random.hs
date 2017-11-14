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
-- Module      : Rand
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


class (PrimMonad m) => Generator g m where
    create :: Maybe Word32 -> m g

instance Generator Mersenne.MTGen IO where
    create = Mersenne.newMTGen

instance Generator (MWC.Gen RealWorld) IO where
    create x = case x of
        (Just n) -> MWC.initialize . singleton $ n
        Nothing  -> MWC.create

instance Generator (MWC.Gen s) (ST s) where
    create x = case x of
        (Just n) -> MWC.initialize . singleton $ n
        Nothing  -> MWC.create


class (Generator g m) => Randomisable a g m where
    random :: g -> m a


instance (Mersenne.MTRandom a) => Randomisable a Mersenne.MTGen IO where
    random g = Mersenne.random g

instance {-# OVERLAPS #-} (Num a, Randomisable a Mersenne.MTGen IO)
  => Randomisable (Complex a) Mersenne.MTGen IO where
    random g = liftM2 (:+) (random g) (random g)

instance (MWC.Variate a) => Randomisable a (MWC.Gen RealWorld) IO where
    random g = MWC.uniform g

instance {-# OVERLAPS #-} (Num a, Randomisable a (MWC.Gen RealWorld) IO)
  => Randomisable (Complex a) (MWC.Gen RealWorld) IO where
    random g = liftM2 (:+) (random g) (random g)


mkMTGen :: Maybe Word32 -> IO Mersenne.MTGen
mkMTGen = create

mkMWCGen :: Maybe Word32 -> IO MWC.GenIO
mkMWCGen = create


class (Generator g m) => UniformDist a g m where
  uniform :: (a, a) -> g -> m a

class (Generator g m) => RandomScalable a g m where
  randScale :: a -> g -> m a


uniformFloating :: (Generator g m, RealFloat a, Randomisable a g m)
                => (a, a) -> g -> m a
uniformFloating (low, high) = random >=> (\r -> return $ low + (high - low) * r)


instance (Generator g m, Randomisable Float g m)
  => UniformDist Float g m where
    uniform = uniformFloating

instance (Generator g m, Randomisable Double g m)
  => UniformDist Double g m where
    uniform = uniformFloating

instance (Generator g m, Randomisable CFloat g m)
  => UniformDist CFloat g m where
    uniform = uniformFloating

instance (Generator g m, Randomisable CDouble g m)
  => UniformDist CDouble g m where
    uniform = uniformFloating


instance (RealFloat a, UniformDist a g m)
  => UniformDist (Complex a) g m where
    uniform ((rlow :+ ilow), (rhigh :+ ihigh)) g =
      liftM2 (:+) (uniform (rlow, rhigh) g) (uniform (ilow, ihigh) g)

instance (UniformDist a g m) => UniformDist [a] g m where
  uniform (low, high) g = mapM (\(l, h) -> uniform (l, h) g) $ zip low high

instance (Foreign.Storable a, UniformDist a g m)
  => UniformDist (V.Vector a) g m where
    uniform (low, high) g = V.zipWithM (\l h -> uniform (l, h) g) low high

randScaleFloating :: (RealFloat a, Generator g m,  Randomisable a g m)
                  => a -> g -> m a
randScaleFloating x = liftM (* x) . random

instance (Generator g m, Randomisable Float g m)
  => RandomScalable Float g m where
    randScale = randScaleFloating

instance (Generator g m, Randomisable Double g m)
  => RandomScalable Double g m where
    randScale = randScaleFloating

instance (Generator g m, Randomisable CFloat g m)
  => RandomScalable CFloat g m where
    randScale = randScaleFloating

instance (Generator g m, Randomisable CDouble g m)
  => RandomScalable CDouble g m where
    randScale = randScaleFloating

instance {-# OVERLAPS #-} (Generator g m, RandomScalable a g m)
  => RandomScalable (Complex a) g m where
    randScale (x :+ y) g = return (:+) `ap` (randScale x g) `ap` (randScale y g)

instance (Generator g m, RandomScalable a g m,
  GV.Vector v a) => RandomScalable (v a) g m where
    randScale x g = GV.mapM (flip randScale g) x


