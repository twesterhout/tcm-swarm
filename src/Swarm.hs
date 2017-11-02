-- {-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
-- {-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TemplateHaskell #-}

{-# OPTIONS_HADDOCK show-extensions #-}


-- |
-- Module      : Swarm
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental


module Swarm
    ( Addable(..)
    , Subtractable(..)
    , Multipliable(..)
    , Randomisable(..)
    , VectorSpace(..)
    , Sized(..)

    , PSOGuide(..)
    , Bee(..)
    , Swarm(..)
    , Updater(..)
    , upScale
    , upLocal
    , upGlobal
    , standardUpdater
    , initSwarm
    , updateBee
    , updateSwarm
    , optimise1D
    , iterateM
    , bees
    , pos
    , vel
    , guide
    , val
    , iteration
    -- , position
    -- , velocity
    -- , value
    -- , best
    -- , initSwarm
    -- , initBee
    ) where

import Data.Monoid
import Control.Monad
import Control.Monad.State
import System.Random
import Data.Maybe
import qualified Data.List as L
import qualified Foreign
import qualified Numeric.LinearAlgebra as LA
import Control.Lens


infixl 7 *&
infixl 6 +&
infixl 6 -&

-- type family ElementOf a :: *

-- | General addition.
class Addable a b c | a b -> c where
    (+&) :: a -> b -> c

-- | General subtraction
class Subtractable a b c | a b -> c where
    (-&) :: a -> b -> c

-- | General multiplication
class Multipliable a b c | a b -> c where
    (*&) :: a -> b -> c

-- | For numeric types @'+&'@ is just @'+'@.
instance {-# OVERLAPPABLE #-} (Num a) => Addable a a a where
    x +& y = x + y

-- | For numeric types @'-&'@ is just @'-'@.
instance {-# OVERLAPPABLE #-} (Num a) => Subtractable a a a where
    x -& y = x - y

-- | For numeric types @'*&'@ is just @'*'@.
instance {-# OVERLAPPABLE #-} (Num a) => Multipliable a a a where
    x *& y = x * y


class Randomisable p where
    randV :: (RandomGen g) => p -> State g p

class ( Num f
      , Addable p p p
      , Subtractable p p p
      , Multipliable f p p
      ) => VectorSpace p f

class Sized p t | p -> t where
    norm :: p -> t

-- type instance ElementOf Double = Double
-- type instance ElementOf Float  = Float
-- type instance ElementOf [a]    = a

instance (Num a, Random a) => Randomisable a where
    randV x = state $ randomR (0, x)

instance VectorSpace Double Double
-- instance PSOVector Double

instance Sized Double Double where
    norm x = abs x

data PSOGuide p r =
    PSOGuide { _psoguidePos :: p
             , _psoguideVal :: r
             } deriving (Show, Eq)

data Bee p r =
    Bee { _beePos :: p
        , _beeVel :: p
        , _beeGuide :: PSOGuide p r
        } deriving (Show, Eq)

data Updater g p r =
    Updater { runUpdater :: Bee p r
                         -> PSOGuide p r
                         -> Int
                         -> State g p
            }

data Swarm g p r =
    Swarm { _swarmBees :: [Bee p r]
          , _swarmGuide :: PSOGuide p r
          , _swarmFunc :: p -> r
          , _swarmRebound :: Bee p r -> Bee p r
          , _swarmUpdater :: Updater g p r
          , _swarmIteration :: Int
          }

makeLensesWith abbreviatedFields ''PSOGuide
makeLensesWith abbreviatedFields ''Bee
makeLensesWith abbreviatedFields ''Swarm

instance (Show p, Show r) => Show (Swarm g p r) where
    show x = "Bees: " ++ show (x^.bees) ++ "\n" ++
             "Guide: " ++ show (x^.guide) ++ "\n" ++
             "Iteration: " ++ show (x^.iteration)

instance Monoid (Updater g p r) where
    mempty = Updater $ \x _ _ -> return (x^.vel)
    mappend a b = Updater $ f
        where f x gg i =
                runUpdater a x gg i >>= \v' ->
                   runUpdater b (x & vel .~ v') gg i

upLocalDynamic :: (RandomGen g, Random a, Randomisable p, VectorSpace p a)
    => (Int -> a) -> Updater g p r
upLocalDynamic c = Updater $ upLocalDynamicImpl c

upLocalDynamicImpl :: (RandomGen g, Random a, Randomisable p, VectorSpace p a)
    => (Int -> a) -> Bee p r -> PSOGuide p r -> Int -> State g p
upLocalDynamicImpl c x _ i = do
    r  <- state random
    dv <- randV $ (x^.guide.pos) -& (x^.pos)
    return $ (x^.vel) +& ((r * (c i)) *& dv)

upLocal :: (RandomGen g, Random a, Randomisable p, VectorSpace p a)
    => a -> Updater g p r
upLocal c  = upLocalDynamic $ const c


upGlobalDynamic :: (RandomGen g, Random a, Randomisable p, VectorSpace p a)
    => (Int -> a) -> Updater g p r
upGlobalDynamic c = Updater $ upGlobalDynamicImpl c

upGlobalDynamicImpl :: (RandomGen g, Random a, Randomisable p, VectorSpace p a)
    => (Int -> a) -> Bee p r -> PSOGuide p r -> Int -> State g p
upGlobalDynamicImpl c x gg i = do
    r  <- state random
    dv <- randV $ (gg^.pos) -& (x^.pos)
    return $ (x^.vel) +& ((r * (c i)) *& dv)

upGlobal :: (RandomGen g, Random a, Randomisable p, VectorSpace p a)
    => a -> Updater g p r
upGlobal c = upGlobalDynamic $ const c


upScaleDynamic :: (RandomGen g, Multipliable a p p)
    => (Int -> a) -> Updater g p r
upScaleDynamic c = Updater $ f c
    where f c x _ i = return $ (c i) *& (x^.vel)

upScale :: (RandomGen g, Multipliable a p p)
    => a -> Updater g p r
upScale c = upScaleDynamic $ const c


upCutoffDynamic :: (RandomGen g, Ord a, Fractional a, Sized p a,
        Multipliable a p p)
    => (Int -> a) -> Updater g p r
upCutoffDynamic c = Updater $ f c
    where f c x _ i = return $
            if (norm (x^.vel)) > (c i)
                then ((c i) / (norm (x^.vel))) *& (x^.vel)
                else (x^.vel)

upCutoff :: (RandomGen g, Ord a, Fractional a, Sized p a,
        Multipliable a p p)
    => a -> Updater g p r
upCutoff c = upCutoffDynamic $ const c


standardUpdater :: (RandomGen g, Random a, Randomisable p,
        VectorSpace p a)
    => (a, a, a) -> Updater g p r
standardUpdater (w, cl, cg) = (upScale w) <> (upLocal cl) <> (upGlobal cg)


updateBee :: (RandomGen g, Addable p p p, Ord r)
    => Swarm g p r -> Bee p r -> State g (Bee p r)
updateBee xs x = do
    v' <- (runUpdater (xs^.updater)) x (xs^.guide) (xs^.iteration)
    let p'     = (x^.pos) +& v'
        value' = (xs^.func) p'
        x'     = if value' < (x^.guide.val)
                   then x & (pos .~ p')
                          & (vel .~ v')
                          & (guide .~ (PSOGuide p' value'))
                   else x & (pos .~ p')
                          & (vel .~ v')
    return $ (xs^.rebound) x'


bestGuide :: (Ord r) => [Bee p r] -> PSOGuide p r
bestGuide = L.minimumBy (compBy (view val)) . map (view guide)
    where compBy f x y = compare (f x) (f y)


updateSwarm :: (RandomGen g, Addable p p p, Ord r)
            => Swarm g p r
            -> State g (Swarm g p r)
updateSwarm swarm = do
    xs <- mapM (updateBee swarm) (swarm^.bees)
    return $ swarm & (bees .~ xs)
                   & (guide .~ (bestGuide xs))
                   & (iteration +~ 1)


iterateM :: (Monad m)
         => (a -> m a)
         -> a
         -> m [a]
iterateM f x = do
    y <- f x
    ys <- iterateM f y
    return $ x : ys


initSwarm :: (RandomGen g, Ord r)
    => Updater g p r
    -> ((p -> r) -> State g (Bee p r))
    -> (Bee p r -> Bee p r)
    -> (p -> r)
    -> Int
    -> State g (Swarm g p r)
initSwarm update initBee rebound func n = do
    xs <- mapM (\_ -> initBee func) $ replicate n ()
    return $ Swarm xs (bestGuide xs) func rebound update 0


optimise1D :: (RandomGen g, Ord r)
    => (Double, Double, Double)
    -> (Double, Double)
    -> (Double -> r)
    -> Int
    -> State g [Swarm g Double r]
optimise1D wpg (low, high) func n =
    let up = standardUpdater wpg
        mkBee f = do
            p <- state $ randomR (low, high)
            return $ Bee p 0 (PSOGuide p (f p))
        mkBounds x =
            if (x^.pos) > high || (x^.pos) < low
                then x & (vel *~ (-1))
                else x
    in do
        swarm <- initSwarm up mkBee mkBounds func n
        iterateM updateSwarm swarm
