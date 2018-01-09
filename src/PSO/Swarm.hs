{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_HADDOCK show-extensions #-}

-- |
-- Module      : Swarm
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.Swarm
  ( -- * Idea
    --
    -- | This module implements different variants of Particle Swarm
    -- Optimisation (PSO), so if you're completely unfamiliar with PSO, go read
    -- https://en.wikipedia.org/wiki/Particle_swarm_optimization first, it's a
    -- nice introduction.
    --
    -- Let there be \(M\in\mathbb{N}\) 'Bee's in the swarm. Bees live in a phase
    -- space (which is a vector space). What this means is that each bee is
    -- described by a phase \(\psi\). The two most common choices of the phase
    -- are:
    --
    -- * \((q, p)\in\mathcal{H}^2\) with \(q\) being the canonical coordinate
    -- and \(p\) being the canonical momentum.
    -- * \(|\psi\rangle\in\mathcal{H}^2\), i.e. a vector in Hilbert space.
    --
    -- While flying around, bees may pick up some information about the
    -- surrounding environment. We call the part of it that bees remember a
    -- local (or private) /guide/. It's a guide because it helps bees decide
    -- where to move next. The simplest and most common guide is 'BeeGuide'.
    --
    -- The whole point of having a swarm of bees rather than one bee is
    -- collective behavior. In case of PSO this is achieved by having a /global
    -- guide/. Each bee then uses this global guide, its own local guide, and
    -- its current phase to compute the next phase. This is represented by
    -- 'PhaseUpdater'.
    --
    -- The whole optimisation process then looks more or less like this:
    --
    --    1. __Initialisation__ (see 'mkSwarm' function).
    --
    --        * Generate initial phases \((\psi^{(0)}_i)_{i\in\{1\dots M\}}\).
    --        * Evaluate fitness in these phases, i.e. compute
    --          \((f(\psi^{(0)}_i))_{i\in\{1\dots M\}}\) (see 'mkBee' function).
    --        * Initialise local guides \((g^{(0)}_{l,i})_{i\in\{1\dots M\}}\)
    --          (see 'mkLocalG' function).
    --        * Construct bees using phases, fitness values, and guides (see
    --          'mkBee' function).
    --        * Initialise global guide \(g^{(0)}_g\) (see 'mkGlobalG'
    --          function).
    --        * Construct swarm using bees and global guide.
    --
    --    2. __Main Loop__. While the termination condition is not met, do the
    --    following (see 'updateSwarm' function):
    --
    --        * Update each bee (see 'updateBee' function), i.e.
    --
    --            * Using global guide \(g^{(n)}_g\), local guide
    --              \(g^{(n)}_{l,i}\), current phase \(\psi^{(n)}_i\), and
    --              \(f(\psi^{(n)}_i)\), come up with a new phase
    --              \(\psi^{(n+1)}_i\) (see 'PhaseUpdater').
    --            * Evaluate the fitness in this new phase, i.e. compute
    --              \(f(\psi^{(n+1)}_i)\).
    --            * Using \(\psi^{(n+1)}_i\) and \(f(\psi^{(n+1)}_i)\), compute
    --              the local guide in the next iteration \(g^{(n+1)}_i\) (see
    --              'updateLocalG' function).
    --
    --        * Update the global guide (see 'updateGlobalG' function).
    --

    -- * Basic types
    Bee(..)
  , CMState(..)
  , QMState(..)
  , LocalGuide(..)
  , BeeGuide(..)
  , GlobalGuide(..)
  , SwarmGuide(..)
  , Swarm(..)
  , PhaseUpdater(..)

    -- * Important functions
  , wpgUpdater
  -- , kickUpdater
  , deltaUpdater
  , absorbUpdater
  , mkCMState
  , mkQMState
  , optimiseND

    -- * Details
  , VelocityUpdater(..)
  , upScale
  , upLocal
  , upGlobal
  , upStep
  , updateBee
  , updateSwarm
  , mkBee
  , mkSwarm
  , fromPure
  , iterateNM
  , iterateWhileM
  , Absorbable(..)
  , DeltaWell(..)
  , Scalable(..)

    -- * Lenses
  , HasPos(..)
  , HasVal(..)
  , HasVar(..)
  , HasGuide(..)
  , HasState(..)
  , HasRun(..)
  , HasBees(..)
  , HasUpdater(..)
  , HasIteration(..)
  ) where

import Data.Complex
import Data.Ord(comparing)
import qualified Data.List as List
-- import Data.Monoid
import Data.Semigroup
import qualified Data.Vector.Generic as GV
import qualified Data.Vector.Storable as V

import Control.Arrow
import Control.Lens
import Control.Monad
import Control.Monad.Primitive
import Control.Monad.Reader

import qualified Foreign
import Foreign.Storable.Tuple

import GHC.Float (float2Double)

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Devel(matrixFromVector, orderOf, MatrixOrder(..))
import Numeric.LinearAlgebra.Data(flatten, tr)
import qualified Numeric.LinearAlgebra.Devel as LADevel

import PSO.VectorSpace
import PSO.Random


-- | A bee in our swarm.
--
-- * @α@ is the type of a point in our phase space (i.e. position-velocity pair
-- in the classical case and wave function in the quantum mechanical)
-- * @β@ is the type of the local guide, i.e. \"memory\" of the bee.
-- * @r@ return type of the function we're minimising.
--
-- /Note:/ use 'state', 'guide', and 'val' lenses to access the fields.
data Bee α β r = Bee
  { _beeState :: !α
  , _beeGuide :: !β
  , _beeVal   :: !r
  } deriving (Show)

makeLensesWith abbreviatedFields ''Bee

-- | Point in the classical phase space.
--
-- @χ@ -- vector in coordinate space. We always use time steps of 1, so physical
-- dimension of velocity is equal to the one of coordinate. The type of velocity
-- is thus also @χ@.
--
-- /Note:/ use 'pos' and 'vel' lenses to access the fields.
data CMState χ = CMState
  { _cmstatePos :: !χ
  , _cmstateVel :: !χ
  } deriving (Show)

makeLensesWith abbreviatedFields ''CMState

-- | Point in our quantum mechanical phase space (i.e. Hilbert space).
--
-- /Note:/ use 'pos' lens to access the field.
data QMState χ = QMState
  { _qmstatePos :: !χ
  } deriving (Show)

makeLensesWith abbreviatedFields ''QMState

-- | Rather than limiting ourselves to a single type of local guide (because who
-- knows which info we might want to keep track of), we define a type class.
-- @LocalGuide β α r@ means that @β@ is a local guide for a bee with phase of
-- type @α@ and fitness function returning @r@.
class LocalGuide β α r where
  -- | Initialisation of a guide. Given a phase \(\psi^{(0)}\) and its fitness
  -- \(f(\psi^{(0)})\), @mkLocalG@ creates local guide \(g^{(0)}_l\).
  mkLocalG     :: α -> r -> β
  -- | Update step. After computing the phase \(\psi^{(n+1)}\) (of type @α@) and
  -- its fitness \(f(\psi^{(n+1)})\) (of type @r@) in the @(n+1)@'st iteration,
  -- we use @updateLocalG@ to compute \(g^{(n+1)}_l\) given \(g^{(n)}_l\).
  updateLocalG :: α -> r -> β -> β

-- | Similar to 'LocalGuide' we define a class of global guides rather than a
-- single one. Here, @GlobalGuide γ β@ means that @γ@ can serve as a global
-- guide for local guides of type @β@.
class (LocalGuide β α r) => GlobalGuide γ β α r where
  -- | Initialisation of a guide. Given a collection of bees, @mkLocalG@
  -- initialises global guide \(g^{(0)}_g\).
  mkGlobalG     :: (Foldable t, Functor t) => t (Bee α β r) -> γ
  -- | Update step. After computing phases, values, and local guides of bees in
  -- the @(n+1)@'st iteration, we use @updateLocalG@ to compute \(g^{(n+1)}_g\)
  -- given \(g^{(n)}_g\).
  updateGlobalG :: (Foldable t, Functor t) => t (Bee α β r) -> γ -> γ

-- | A guide that remembers position and its corresponding fitness.
--
-- /Note:/ use 'pos' and 'val' lenses to access the fields.
data BeeGuide χ r = BeeGuide
  { _beeguidePos :: !χ -- ^ Position.
  , _beeguideVal :: !r -- ^ Fitness value.
  } deriving (Show)

makeLensesWith abbreviatedFields ''BeeGuide

-- | We compare guides by their fitness values.
instance Eq r => Eq (BeeGuide χ r) where
  x == y = (x ^. val) == (y ^. val)

-- | We compare guides by their fitness values.
instance Ord r => Ord (BeeGuide χ r) where
  compare = comparing (view val)

-- | 'BeeGuide' is the simplest and, probably, the most useful example of a
-- 'LocalGuide'. It keeps track of the best position and its fitness. That is,
-- if we've run @n@ iterations (calls to 'updateLocalG'), then the best position
-- is given by \(\operatorname{argmin}_{k\in\{0\dots n\}}f(\psi^{(k)}_i)\), and
-- its fitness is just \(\operatorname{min}_{k\in\{0\dots n\}}f(\psi^{(k)}_i)\).
instance (HasPos α χ, Ord r) => LocalGuide (BeeGuide χ r) α r where
  mkLocalG ψ fψ = BeeGuide (ψ ^. pos) fψ
  updateLocalG ψ fψ g
    | fψ < g ^. val = g & (pos .~ (ψ ^. pos))
                        & (val .~ fψ)
    | otherwise = g

-- | 'BeeGuide' can also serve as a 'GlobalGuide'. It keeps track of the overall
-- best position among all bees and iterations and its fitness. So after @n@
-- iterations, the best position is given by
-- \(\operatorname{argmin}_{k\in\{0\dots n\},i\in\{1\dots M\}}f(\psi^{(k)}_i)\),
-- and its fitness is
-- \(\operatorname{min}_{k\in\{0\dots n\},i\in\{1\dots M\}}f(\psi^{(k)}_i)\).
instance (LocalGuide (BeeGuide χ r) α r, Ord (BeeGuide χ r))
  => GlobalGuide (BeeGuide χ r) (BeeGuide χ r) α r where
    mkGlobalG = minimum . fmap (view guide)
    updateGlobalG xs _ = minimum . fmap (view guide) $ xs

-- | A guide that remembers
--
--    * Best position throughout the whole swarm.
--    * Fitness value at this position.
--    * Current iteration.
--    * Variance in the fitness values of the local guides.
--
-- /Note:/ use 'pos', 'val', 'iteration', and 'var' lenses to access the fields.
data SwarmGuide χ r = SwarmGuide
  { _swarmguidePos       :: !χ
  , _swarmguideVal       :: !r
  , _swarmguideIteration :: !Int
  , _swarmguideVar       :: !r
  } deriving (Show)

makeLensesWith abbreviatedFields ''SwarmGuide

-- | Calculates the mean value of a sequence.
mean :: (Fractional a, Foldable v) => v a -> a
mean xs = sum xs / fromIntegral (length xs)

-- | Calculates the variance in a sequence.
variance :: (Fractional a, Foldable v, Functor v) => v a -> a
variance xs = (mean . fmap (^^2) $ xs) - (mean xs)^^2

-- | 'SwarmGuide' is another example of a 'GlobalGuide'. On top of keeping track
-- of the best position and its fitness value (what 'BeeGuide' does), we
-- remember the current iteration ('mkGlobalG' sets it to @0@, and each call to
-- 'updateGlobalG' increases the iteration by @1@). And finally, each iteration
-- we compute the variance \(\operatorname{Var}\{f(\psi^{(n)}_i)\}_i\) in the
-- fitness values of bees.
instance (LocalGuide (BeeGuide χ r) α r, Ord (BeeGuide χ r), Fractional r)
  => GlobalGuide (SwarmGuide χ r) (BeeGuide χ r) α r where
    mkGlobalG xs = SwarmGuide (x ^. pos) (x ^. val) 0 var
      where x   = minimum . fmap (view guide) $ xs
            var = variance . fmap (view val) $ xs
    updateGlobalG xs g = mkGlobalG xs & iteration .~ (g ^. iteration + 1)

-- | Given global guide and bee in the @n@'th iteration, calculates the phase of
-- the bee in the @(n+1)@'st iteration. This calculation is done in some 'Monad'
-- @m@ which allows us to, for example, use random numbers when updating the
-- phase.
--
-- /Note:/ use 'run' lens to unpack the function.
newtype PhaseUpdater m γ β α r = PhaseUpdater
  { _phaseupdaterRun :: γ -> Bee α β r -> m α
  }

makeLensesWith abbreviatedFields ''PhaseUpdater

-- | 'PhaseUpdater's can be combined to build more difficult ones.
instance (Monad m) => Semigroup (PhaseUpdater m γ β α r) where
  (PhaseUpdater a) <> (PhaseUpdater b) = PhaseUpdater update
    where update guide bee = a guide bee >>= \ψ ->
            b guide (bee & state .~ ψ)

-- | Our swarm.
data Swarm m γ β α r = Swarm
  { _swarmBees      :: ![Bee α β r] -- ^ Collection of bees.
  , _swarmGuide     :: !γ -- ^ Global guide.
  , _swarmFunc      :: !(α -> m r) -- ^ Fitness function.
  , _swarmUpdater   :: !(PhaseUpdater m γ β α r) -- ^ Updater.
  }

makeLensesWith abbreviatedFields ''Swarm

-- | Updates the canonical momentum in the classical phase. Given a global guide
-- and a bee described by classical mechanics, returns the momentum of the bee
-- in the next iteration. 'VelocityUpdater's can be combined using '<>' operator
-- and afterwards converted to 'PhaseUpdater's using 'upStep' function.
newtype VelocityUpdater m γ χ r = VelocityUpdater
  { _velocityupdaterRun :: γ -> Bee (CMState χ) (BeeGuide χ r) r -> m χ
  }

makeLensesWith abbreviatedFields ''VelocityUpdater

-- | 'VelocityUpdater's can be combined to form more complex ones.
instance (Monad m) => Semigroup (VelocityUpdater m γ χ r) where
  (VelocityUpdater a) <> (VelocityUpdater b) = VelocityUpdater $ update
    where update guide bee =
            a guide bee >>= \v -> b guide (bee & (state . vel) .~ v)

-- | Given a constant \(\varphi_l\), creates a 'VelocityUpdater' that updates
-- momentum according to
-- \[
--    p^{(n+1)} = p^{(n)} + \varphi_l U(0, 1) \circ (g^{(n)}_l - q^{(n)}) \;,
-- \]
-- where \(\circ\) denotes the Hadamard product, \(U(0, 1)\) is a vector
-- whose elements are uniformly distributed in the \([0, 1)\) interval, and
-- \(g^{(n)}_l\) is local best position in the @n@'th iteration.
upLocal :: (RandomScalable m χ, VectorSpace χ λ) => λ -> VelocityUpdater m γ χ r
upLocal φl = VelocityUpdater $ upLocalImpl φl

upLocalImpl ::
     (RandomScalable m χ, VectorSpace χ λ, HasPos α χ, HasVel α χ, HasPos β χ)
  => λ -> γ -> Bee α β r -> m χ
upLocalImpl φ _ bee =
  let δq = bee ^. guide . pos - bee ^. state . pos
      p  = bee ^. state . vel
   in liftM (scale φ) (randScale δq) >>= \δp -> return (p + δp)

-- | Given a constant \(\varphi_g\), creates a 'VelocityUpdater' that updates
-- momentum according to
-- \[
--    p^{(n+1)} = p^{(n)} + \varphi_g U(0, 1) \circ (g^{(n)}_g - q^{(n)}) \;,
-- \]
-- where \(\circ\) denotes the Hadamard product, \(U(0, 1)\) is a vector
-- whose elements are uniformly distributed in the \([0, 1)\) interval, and
-- \(g^{(n)}_g\) is global best position in the @n@'th iteration.
upGlobal ::
     (RandomScalable m χ, VectorSpace χ λ, HasPos γ χ)
  => λ -> VelocityUpdater m γ χ r
upGlobal φg = VelocityUpdater $ upGlobalImpl φg

upGlobalImpl ::
     (RandomScalable m χ, VectorSpace χ λ, HasPos α χ, HasVel α χ, HasPos γ χ)
  => λ -> γ -> Bee α β r -> m χ
upGlobalImpl φ g bee = do
  let δq = g ^. pos - bee ^. state . pos
      p  = bee ^. state . vel
   in liftM (scale φ) (randScale δq) >>= \δp -> return (p + δp)

-- | Given a constant \(\omega\), creates a 'VelocityUpdater' that updates
-- momentum according to
-- \[
--    p^{(n+1)} = \omega p^{(n)} \;.
-- \]
upScale :: (Monad m, Scalable λ χ) => λ -> VelocityUpdater m γ χ r
upScale ω = VelocityUpdater $ upScaleImpl ω

upScaleImpl :: (Monad m, Scalable λ χ, HasVel α χ) => λ -> γ -> Bee α β r -> m χ
upScaleImpl ω _ = return . scale ω . view (state . vel)

-- | Converts a 'VelocityUpdater' into a 'PhaseUpdater'. This is done by first
-- running the velocity updater to compute \(p^{(n+1)}\). \(q^{(n+1)}\) is then
-- calculated as \(q^{(n)} + p^{(n+1)}\).
upStep ::
     (Monad m, Num χ)
  => VelocityUpdater m γ χ r -> PhaseUpdater m γ (BeeGuide χ r) (CMState χ) r
upStep (VelocityUpdater upVel) = PhaseUpdater update
  where update guide bee = do
          δq <- upVel guide bee
          let x = bee ^. state
              q = x ^. pos
          return $ x & (vel .~ δq)
                     & (pos .~ q + δq)

-- | Creates the standard \"WPG\" updater. Given constants
-- \(\omega,\varphi_l,\varphi_g\), creates a 'PhaseUpdater' that updates the
-- phase according to
-- \[
--  \left\{
--  \begin{aligned}
--    p^{(n+1)} &= \omega p^{(n)}
--              + \varphi_l U_l(0, 1) \circ (g^{(n)}_l - q^{(n)})
--              + \varphi_g U_g(0, 1) \circ (g^{(n)}_g - q^{(n)}) \;, \\
--    q^{(n+1)} &= q^{(n)} + p^{(n+1)} \;,
--  \end{aligned}
--  \right.
-- \]
-- where \(\circ\) denotes the Hadamard product and \(U_l(0, 1)\) and
-- \(U_g(0,1)\) are vectors whose elements are uniformly distributed in the
-- \([0, 1)\) interval.
wpgUpdater ::
     (RandomScalable m χ, VectorSpace χ λ, HasPos γ χ)
  => (λ, λ, λ) -> PhaseUpdater m γ (BeeGuide χ r) (CMState χ) r
wpgUpdater (ω, φl, φg) =
  upStep $ (upScale ω) <> (upLocal φl) <> (upGlobal φg)


-- | Creates the standard updater which uses constriction rather than inertia
-- parameter. Given constants
-- \(\kappa,\varphi_l,\varphi_g\), creates a 'PhaseUpdater' that updates the
-- phase according to
-- \[
--  \left\{
--  \begin{aligned}
--    p^{(n+1)} &= \kappa \left( p^{(n)}
--              + \varphi_l U_l(0, 1) \circ (g^{(n)}_l - q^{(n)})
--              + \varphi_g U_g(0, 1) \circ (g^{(n)}_g - q^{(n)}) \right) \;, \\
--    q^{(n+1)} &= q^{(n)} + p^{(n+1)} \;,
--  \end{aligned}
--  \right.
-- \]
-- where \(\circ\) denotes the Hadamard product and \(U_l(0, 1)\) and
-- \(U_g(0,1)\) are vectors whose elements are uniformly distributed in the
-- \([0, 1)\) interval.
constrictedUpdater ::
     (RandomScalable m χ, VectorSpace χ λ, HasPos γ χ)
  => (λ, λ, λ) -> PhaseUpdater m γ (BeeGuide χ r) (CMState χ) r
constrictedUpdater (κ, φl, φg) =
  upStep $ (upLocal φl) <> (upGlobal φg) <> (upScale κ)

-- randomWalkUpdater :: (UniformDist p gen m)
--                   => (p, p) -> Updater m gen s g p r
-- randomWalkUpdater bounds = Updater $ f
--   where f _ _ = do
--           randGen <- ask
--           v' <- lift $ uniform bounds randGen
--           return v'
-- 
-- kickUpdater ::
--      ( RandomScalable p gen m
--      , UniformDist p gen m
--      , VectorSpace p a
--      , Scalable r p
--      , HasVar s r
--      , HasPos s p
--      , HasPos g p
--      , Floating r
--      , Ord r
--      , s ~ SwarmGuide p r
--      )
--   => (r, r) -> (p, p) -> (a, a, a) -> Updater m gen s g p r
-- kickUpdater cutoff bounds wpg = Updater $ kickUpdaterImpl cutoff bounds wpg
-- 
-- kickUpdaterImpl ::
--      ( RandomScalable p gen m
--      , UniformDist p gen m
--      , VectorSpace p a
--      , Scalable r p
--      , HasVar s r
--      , HasPos s p
--      , HasPos g p
--      , Floating r
--      , Ord r
--      )
--   => (r, r) -> (p, p) -> (a, a, a) -> s -> Bee p r g -> ReaderT gen m p
-- kickUpdaterImpl (cutoff, c) bounds wpg stats x = runUpdater updater stats x
--   where updater = if (stats ^. var) < cutoff
--                      then randomWalkUpdater bounds
--                           <> upScale (c * sqrt (stats ^.var))
--                      else standardUpdater wpg


-- | Update the position according to QDPSO (Quantum Delta well Particle Swarm
-- Optimisation).
class (Monad m) => DeltaWell m λ χ where
  upDeltaWell :: λ -> χ -> χ -> m χ

-- | 1D QDPSO updater. Given a constant \(\kappa\in\mathbb{R}\), center
-- \(g^{(n)}\in\mathbb{R}\) of the density distribution, and bees current
-- position \(q^{(n)}\in\mathbb{R}\), its position in the next iteration is
-- computed as
-- \[
--  q^{(n+1)} =
--    q^{(n)} + \sgn(2\cdot\operatorname{rand}(0,1) - 1) \kappa
--                \log(\frac{1}{\operatorname{rand}(0, 1)}) \cdot
--                  |q^{(n)} - g^{(n)}| \;.
-- \]
instance (Monad m, RealFloat λ, Ord λ, Randomisable m λ)
  => DeltaWell m λ λ where
    upDeltaWell κ p x = do
      u    <- random
      sign <- (\c -> signum (2 * c - 1)) <$> random
      return $ p + (sign * (log . recip $ u) / κ) * abs (x - p)

-- | Extending QDPSO to \(\mathbb{C}\). 1D real version is simply applied to
-- both real and imaginary parts.
instance (RealFloat λ, DeltaWell m λ λ)
  => DeltaWell m λ (Complex λ) where
    upDeltaWell κ p x =
      let f getter = uncurry (upDeltaWell κ) . over each getter
       in return (:+) `ap` f realPart (p, x)
                      `ap` f imagPart (p, x)

-- | Extending QDPSO to multidimensional case. 1D version is simply applied to
-- each component.
instance (Monad m, Foreign.Storable ξ, DeltaWell m λ ξ)
  => DeltaWell m λ (V.Vector ξ) where
    upDeltaWell κ p x = V.zipWithM (upDeltaWell κ) p x

liftMatrix2M ::
     (Monad m, LA.Element α, LA.Element β, LA.Element γ,
      LA.Container V.Vector α, Num α,
      LA.Container V.Vector β, Num β,
      LA.Transposable (LA.Matrix α) (LA.Matrix α),
      LA.Transposable (LA.Matrix β) (LA.Matrix β))
  => (LA.Vector α -> LA.Vector β -> m (LA.Vector γ))
  -> LA.Matrix α -> LA.Matrix β -> m (LA.Matrix γ)
liftMatrix2M f m1@(LA.size -> (r,c)) m2
  | LA.size m1 /= LA.size m2 = error "nonconformant matrices in liftMatrix2M"
  | orderOf m1 == RowMajor =
      return (matrixFromVector RowMajor r c) `ap` f (flatten m1) (flatten m2)
  | otherwise =
      return (matrixFromVector ColumnMajor r c) `ap`
        f (flatten . tr $ m1) (flatten . tr $ m2)

instance (Monad m, Num ξ, LA.Container V.Vector ξ, LA.Transposable (LA.Matrix ξ)(LA.Matrix ξ), DeltaWell m λ ξ)
  => DeltaWell m λ (LA.Matrix ξ) where
    upDeltaWell κ = liftMatrix2M (V.zipWithM (upDeltaWell κ))


-- | Creates a QDPSO updater.
deltaUpdater :: forall m g γ β χ λ r.
     ( Randomisable m λ
     , RealFloat λ
     , Ord λ
     , VectorSpace χ λ
     , DeltaWell m λ χ
     , HasPos β χ
     , HasPos γ χ
     )
  => λ -> γ -> Bee (QMState χ) β r -> m (QMState χ)
deltaUpdater κ g bee = do
  φ₁ <- random :: m λ
  φ₂ <- random :: m λ
  -- lift $ putStrLn $ "φ₁ = " ++ show φ₁ ++ ", φ₂ = " ++ show φ₂
  let center = (φ₁ / (φ₁ + φ₂)) `scale` (g ^. pos)
             + (φ₂ / (φ₁ + φ₂)) `scale` (bee ^. guide . pos)
  x <- upDeltaWell κ center (bee ^. state . pos)
  -- lift $ putStrLn $ "p - x = " ++ show (bee ^. state . pos - p)
  return $ (bee ^. state) & (pos .~ x)



data RelativePosition = Inside | LeftOf | RightOf

class RelativelyPositioned α β where
  getRelPos :: α -> β -> RelativePosition

instance (Ord λ) => RelativelyPositioned (λ, λ) λ where
  getRelPos (l, h) x
    | x < l     = LeftOf
    | x > h     = RightOf
    | otherwise = Inside

class Absorbable λ α where
  absorb :: (λ, λ) -> α -> α

class Reflectable λ α where
  reflect :: (λ, λ) -> α -> α

instance (Ord χ) => Absorbable χ χ where
  absorb (l, h) x
    | x < l     = l
    | x > h     = h
    | otherwise = x

instance (Ord λ) => Absorbable λ (Complex λ) where
  absorb bs = over each (absorb bs)

instance (Ord λ, Num λ)
  => Absorbable λ (λ, λ) where
    absorb (l, h) (q, p) = case getRelPos (l, h) q of
                             Inside  -> (q, p)
                             LeftOf  -> (l, 0)
                             RightOf -> (h, 0)

zipWithTuple :: (a -> b -> c) -> (a, a) -> (b, b) -> (c, c)
zipWithTuple f (x₁, x₂) (y₁, y₂) = (f x₁ y₁, f x₂ y₂)

instance (Ord λ, Num λ)
  => Absorbable λ (Complex λ, Complex λ) where
    absorb bounds ψ =
      let f getter = absorb bounds . over each getter
       in zipWithTuple (:+) (f realPart ψ) (f imagPart ψ)

instance (Absorbable λ (ξ, ξ), GV.Vector χ ξ, GV.Vector χ (ξ, ξ))
  => Absorbable λ (CMState (χ ξ)) where
    absorb bounds ψ =
      let (q, p) = GV.unzip . GV.map (absorb bounds)
                 $ GV.zip (ψ ^. pos) (ψ ^. vel)
       in ψ & (pos .~ q) & (vel .~ p)

instance (Absorbable λ ξ, GV.Vector χ ξ)
  => Absorbable λ (QMState (χ ξ)) where
    absorb bounds = pos %~ GV.map (absorb bounds)

absorbUpdater :: (Monad m, Absorbable λ α) => (λ, λ) -> PhaseUpdater m γ β α r
absorbUpdater bounds = PhaseUpdater $ upAbsorbImpl bounds

upAbsorbImpl :: (Monad m, Absorbable λ α) => (λ, λ) -> γ -> Bee α β r -> m α
upAbsorbImpl bounds _ = return . absorb bounds . view state


-- | Given a swarm and bee in the @n@'th iteration, returns a computation that
-- returns the bee in the @(n+1)@'st iteration.
updateBee ::
     (Monad m, LocalGuide β α r)
  => Swarm m γ β α r -> Bee α β r -> m (Bee α β r)
updateBee xs x = do
  ψ'  <- (xs ^. updater . run) (xs ^. guide) x
  fψ' <- (xs ^. func) ψ'
  return $ x & (state .~ ψ')
             & (val   .~ fψ')
             & (guide %~ updateLocalG ψ' fψ')


-- | Given a swarm in the @n@'th iteration, returns a computation that returns
-- the swarm in the @(n+1)@'st iteration.
updateSwarm :: (Monad m, GlobalGuide γ β α r)
            => Swarm m γ β α r -> m (Swarm m γ β α r)
updateSwarm swarm = do
  xs <- mapM (updateBee swarm) (swarm ^. bees)
  return $ swarm & (bees .~ xs)
                 & (guide %~ updateGlobalG xs)

-- | Applies \(f\) to \(x\) \(\max(0, n - 1)\) times and accumulates the
-- results. Returned list will have \(n\) elements.
iterateNM ::
     (Monad m)
  => Int -- ^ \(n\), number of times to apply \(f\)
  -> (a -> m a) -- ^ function \(f\) to apply
  -> a -- ^ initial value \(x\)
  -> m [a]
iterateNM n f x
  | n == 0 = return [x]
  | otherwise = do
    y <- f x
    ys <- iterateNM (n - 1) f y
    return $ x : ys

iterateWhileM ::
     (Monad m)
  => (a -> Bool)
  -> (a -> m a) -- ^ function \(f\) to apply
  -> a -- ^ initial value \(x\)
  -> m [a]
iterateWhileM predicate func x
  | predicate x = do
      y  <- func x
      ys <- iterateWhileM predicate func y
      return $ x : ys
  | otherwise   = return [x]


mkCMState :: (UniformDist m χ, Num χ) => (χ, χ) -> m (CMState χ)
mkCMState bounds = liftM (\q -> CMState q 0) $ uniform bounds

mkQMState :: (UniformDist m χ, Num χ) => (χ, χ) -> m (QMState χ)
mkQMState bounds = liftM (\q -> QMState q)   $ uniform bounds


-- | @mkSwarm updater initialiser inside func n@ creates a new 'Swarm' where
--
-- * @updater@ is the update strategy used to obtain velocities of bees in the
-- next iteration.
-- * @initialiser@ creates a new random 'Bee'. It is assumed that position of a
-- bee created using @initialiser@ will satisfy @inside@.
-- * Given a position \(x\in\mathcal{H}\) returns whether \(x\in\mathcal{V}\)
-- holds.
-- * @func@ is the function \(f\) we're trying to minimise.
-- * @n@ is number of bees in the newly created swarm.
mkSwarm ::
     (Monad m, GlobalGuide γ β α r)
  => m α
  -> PhaseUpdater m γ β α r
  -> (α -> m r)
  -> Int
  -> m (Swarm m γ β α r)
mkSwarm newState updater func n = (sequence . replicate n $ newState) >>=
  mapM (mkBee func) >>= \xs ->
    return $ Swarm xs (mkGlobalG xs) func updater

fromPure :: (Monad m, HasPos α χ) => (χ -> r) -> α -> m r
fromPure f ψ = return $ f (ψ ^. pos)
-- 
-- class Between a b where
--   isBetween :: (a, a) -> b -> Bool
-- 
-- instance (Ord a) => Between a a where
--   isBetween (l, h) x = l <= x && x <= h
-- 
-- instance {-# OVERLAPS #-} (Between a a) => Between (Complex a) (Complex a) where
--   isBetween (rl :+ il, rh :+ ih) (rx :+ ix) =
--     isBetween (rl, rh) rx && isBetween (il, ih) ix
-- 
-- instance {-# OVERLAPS #-} (Foreign.Storable a, Between a a)
--   => Between (V.Vector a) (V.Vector a) where
--     isBetween (low, high) x =
--       V.and $ V.zipWith3 (\l h x -> isBetween (l, h) x) low high x
-- 
-- -- | Creates a new 'Bee' with a random position inside a given 1D interval and
-- -- zero velocity.
-- randomBee1D ::
--      (Num p, UniformDist p gen m)
--   => (p, p) -- ^ @[low, high]@ interval. It is assumed that @low <= high@.
--   -> (p -> ReaderT gen m r) -- ^ Function we're minimising.
--   -> ReaderT gen m (Bee p r (BeeGuide p r))
-- randomBee1D bounds func = do
--   g <- ask
--   x <- lift $ uniform bounds g
--   fx <- func x
--   return $ Bee x 0 fx (BeeGuide x fx)


-- | Given a fitness function \(f\) and phase \(\psi^{(0)}\), creates a new
-- 'Bee'.
mkBee :: (Monad m, LocalGuide β α r)
  => (α -> m r) -> α -> m (Bee α β r)
mkBee f ψ = do
  fψ <- f ψ
  return $ Bee ψ (mkLocalG ψ fψ) fψ

-- -- | Simplified @mkSwarm@ for 1D pure functions.
-- -- simpleInit1D ::
-- --      ( Num a
-- --      , RandomScalable a gen m
-- --      , UniformDist a gen m
-- --      , Between a a
-- --      , Ord r
-- --      , s ~ g
-- --      , g ~ BeeGuide a r
-- --      )
-- --   => (a, a, a) -- ^ WPG parameters \((\omega,\varphi_l,\varphi_g)\).
-- --   -> (a, a) -- ^ Boundaries \([a, b]\).
-- --   -> (a -> r) -- ^ Function \(f\) we're minimising.
-- --   -> Int -- ^ Number of bees.
-- --   -> ReaderT gen m (Swarm m gen s g a r)
-- -- simpleInit1D wpg (low, high) func =
-- --   let updater = standardUpdater wpg
-- --       initialiser = randomBee1D (low, high)
-- --       boundaries = isBetween (low, high)
-- --   in mkSwarm updater initialiser boundaries (fromPure func)
-- 
-- 
-- unpackToV :: (Foreign.Storable a) => [(a, a)] -> (V.Vector a, V.Vector a)
-- unpackToV xs = let a = V.fromList . map fst $ xs
--                    b = V.fromList . map snd $ xs
--                 in (a, b)
-- 
-- -- | Simplified @mkSwarm@ for N-dimensional functions.
-- simpleInitND ::
--      ( Foreign.Storable a
--      , VectorSpace p a
--      , RandomScalable p gen m
--      , UniformDist p gen m
--      , Between p p
--      , Ord r
--      , p ~ V.Vector a
--      , GlobalGuide s g p r
--      )
--   => Updater m gen s g p r -- ^ PWG parameters \((\omega,\varphi_l,\varphi_g)\)
--   -> [(a, a)]
--   -> [(a, a)] -- ^ List of intervals \([a_i,b_i]\). \(\mathcal{V}\) is then
--   -- \(\prod_i [a_i,b_i]\).
--   -> (p -> r) -- ^ Function \(f\) we're minimising.
--   -> Int -- ^ Number of bees.
--   -> ReaderT gen m (Swarm m gen s g p r)
-- simpleInitND updater initBounds bounds func =
--   let initialiser = randomBeeND (unpackToV initBounds)
--       boundaries = isBetween (unpackToV bounds)
--   in mkSwarm updater initialiser boundaries (fromPure func)
-- 
-- -- optimise1D ::
-- --      ( Num a
-- --      , RandomScalable a gen m
-- --      , UniformDist a gen m
-- --      , Between a a
-- --      , Ord r
-- --      , s ~ g
-- --      , g ~ BeeGuide a r
-- --      )
-- --   => (a, a, a)
-- --   -> (a, a)
-- --   -> (a -> r)
-- --   -> Int
-- --   -> (Swarm m gen s g a r -> Bool)
-- --   -> ReaderT gen m [Swarm m gen s g a r]
-- -- optimise1D wpg bounds func n predicate = do
-- --   swarm <- simpleInit1D wpg bounds func n
-- --   iterateWhileM (not . predicate) updateSwarm swarm

optimiseND ::
     ( Monad m
     , HasPos α χ
     , GlobalGuide γ β α r
     )
  => m α
  -> PhaseUpdater m γ β α r
  -> (α -> m r)
  -> Int
  -> (Swarm m γ β α r -> Bool)
  -> m [Swarm m γ β α r]
optimiseND newState updater func n predicate = do
  swarm <- mkSwarm newState updater func n
  iterateWhileM (not . predicate) updateSwarm swarm


