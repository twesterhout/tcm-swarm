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
import qualified Data.Vector.Generic as GV

class Scalable a v where
  scale :: a -> v -> v

instance (Num a) => Scalable a a where
  scale = (*)

instance (Num a, GV.Vector v a) => Scalable a (v a) where
  scale c = GV.map (c*)

instance {-# OVERLAPS #-} (Num a) => Scalable a (Complex a) where
  scale c (x :+ y) = (c * x) :+ (c * y)

instance (RealFloat a, GV.Vector v (Complex a))
  => Scalable a (v (Complex a)) where
    scale c = GV.map ((c :+ 0)*)

class (Num a, Num v, Scalable a v) => VectorSpace v a

instance (Num a, Num v, Scalable a v) => VectorSpace v a
