{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_HADDOCK show-extensions #-}

-- |
-- Module      : VectorSpace
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental

module PSO.VectorSpace
  (
    Scalable(..)
  , VectorSpace(..)
  ) where

import           Data.Complex        (Complex (..))
import Foreign.Storable
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Generic as GV
import qualified Numeric.LinearAlgebra as LA

class Scalable a v where
  scale :: a -> v -> v

instance (Num a) => Scalable a a where
  scale = (*)

instance (Num a) => Scalable a (Complex a) where
  scale c (x :+ y) = (c * x) :+ (c * y)

instance (Num a, Storable a) => Scalable a (V.Vector a) where
  scale c = V.map (c*)

instance (RealFloat a, Storable a) => Scalable a (V.Vector (Complex a)) where
  scale c = scale (c :+ 0)

instance (RealFloat a, LA.Container LA.Vector a)
  => Scalable a (LA.Matrix a) where
    scale c = LA.scale c

instance (RealFloat a, LA.Container LA.Vector (Complex a))
  => Scalable a (LA.Matrix (Complex a)) where
    scale c = LA.scale (c :+ 0)

class (Num a, Num v, Scalable a v) => VectorSpace v a

instance (Num a, Num v, Scalable a v) => VectorSpace v a
