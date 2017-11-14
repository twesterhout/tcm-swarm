{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_HADDOCK show-extensions #-}

-- {-# LANGUAGE OverlappingInstances   #-}
-- |
-- Module      : Swarm
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.Swarm
    -- * Idea
    --
    -- | If you're completely unfamiliar with PSO, go read
    -- https://en.wikipedia.org/wiki/Particle_swarm_optimization, it's a nice
    -- introduction.
    --
    -- Let \(M\in\mathbb{N}\) be the number of bees in the swarm. Bees live in
    -- a vector space \(\mathcal{H}\). Each bee has some position
    -- \(x\in\mathcal{H}\) and velocity \(v\in\mathcal{H}\). Task of this swarm
    -- is to minimise \(f: \mathcal{V} \to R\), where
    -- \(\mathcal{V}\subseteq\mathcal{H}\) is /connected/ and \((R,\leq)\) is an
    -- ordered set, i.e. we're looking for
    -- \(\operatorname{arg}\min_{x\in\mathcal{V}}f(x)\).
    -- To do that each bee keeps track of the best place (/local best/)
    -- \(p\in\mathcal{V}\) it has visited so far. The swarm, in turn, keeps
    -- track of the /global best/ \(g\in\mathcal{V}\) among all the bees. Bees
    -- then use this information to decide which way to move next. I.e. we have
    -- a function
    -- \[
    --    \begin{aligned}
    --      &\operatorname{updater}:(\mathcal{V}, \mathcal{V}, \mathcal{V})
    --        \to \mathcal{V} \to \mathcal{V} \;,\\
    --      &\operatorname{updater}:(x^{(n)},v^{(n)},p^{(n)})
    --        \mapsto g^{(n)} \mapsto v^{(n+1)} \;,
    --    \end{aligned}
    -- \]
    -- where superscript \({}^(n)\) means \"n'th iteration\". Next position
    -- \(x^{(n+1)}\) is then computed as \(x^{(n+1)}=x^{(n)}+v^{(n+1)}\).
    -- There's one caveat though: with this definition, there's no guarantee
    -- that \(x^{(n+1)}\in\mathcal{V}\). To work around this issue, we introduce
    -- reflection, i.e.
    -- \[
    --    \begin{aligned}
    --      x^{(n+1)} &= \left\{
    --        \begin{aligned}
    --          & x^{(n)} + v' \;, & x^{(n)} + v' \in    \mathcal{V} \\
    --          & x^{(n)}      \;, & x^{(n)} + v' \notin \mathcal{V}
    --        \end{aligned} \right. \\
    --      v^{(n+1)} &= \left\{
    --        \begin{aligned}
    --          & v'  \;, & x^{(n)} + v' \in    \mathcal{V} \\
    --          & -v' \;, & x^{(n)} + v' \notin \mathcal{V}
    --        \end{aligned} \right. \\
    --      & v' = \operatorname{updater}((x^{(n)}, v^{(n)}, p^{(n)}), g^{(n)})
    --    \end{aligned}
    -- \]
    -- That is, rather than going into \"forbidden\" area
    -- \(\mathcal{H}\setminus\mathcal{V}\), a bee remains at the same place but
    -- reverses its velocity.
    --
    -- We start with some initial random configuration
    -- \(\{x_i^{(0)}\}_{0\leq i\leq M-1} \subset \mathcal{V}\) of positions and
    -- zero velocities. Using the steps above, we construct sequence
    -- \(\{g^{(n)}\}_{n\in\mathbb{N}} \subset \mathcal{V}\) and /hope/ that it
    -- converges to some \(g \in \mathcal{V}\).
    -- * Back to Haskell
    -- ** Basic types
  ( Bee(..)
  , BeeGuide(..)
  , SwarmGuide(..)
  , Updater(..)
  , Swarm(..)
  -- , Addable(..)
  -- , Subtractable(..)
  -- , Multipliable(..)
  , VectorSpace(..)
    -- ** Updater magic
  , upScale
  , upLocal
  , upGlobal
  , standardUpdater
  , standardUpdaterSimple
  , kickUpdater
  , mkSwarm
  , updateBee
  , updateSwarm
  -- , randomBee1D
  , randomBeeND
  -- , simpleInit1D
  , simpleInitND
  -- , optimise1D
  , optimiseND
  , iterateNM
  , iterateWhileM
  , bees
  , pos
  , vel
  , guide
  , stats
  , val
  , var
  , iteration
  , unpackToV
    -- , position
    -- , velocity
    -- , value
    -- , best
    -- , initSwarm
    -- , initBee
  , Between(..)
  , fromPure

  , HasPos
  , HasVal
  , HasVar
  ) where

import Data.Complex
import qualified Data.List as List
import Data.Monoid
import qualified Data.Vector.Generic as GV
import qualified Data.Vector.Storable as V

import Control.Lens
import Control.Monad
import Control.Monad.Primitive
import Control.Monad.Reader

import qualified Foreign
import Foreign.Storable.Tuple

import GHC.Float (float2Double)

import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Devel as LADevel

import PSO.VectorSpace
import PSO.Random

-- | \"Memory\" of a 'Bee' or 'Swarm', i.e. what it remembers about previous
-- iterations. As explained above, we keep track of the best place visited so
-- far. It is accessible through 'pos' lens. We furthermore remember the value
-- of \(f\) in this position (accessible through 'val' lens).
data BeeGuide p r = BeeGuide
  { _psoguidePos :: !p -- ^ \(\in\mathcal{V}\), /best/ place visited so far.
  , _psoguideVal :: !r -- ^ \(\in R\), /best/ value, i.e. value of the function
                       -- \(f\) in the best position.
  } deriving (Show, Eq)

makeLensesWith abbreviatedFields ''BeeGuide

data SwarmGuide p r = SwarmGuide
  { _globalguidePos :: !p
  , _globalguideVal :: !r
  , _globalguideVar :: !r
  } deriving (Show, Eq)

makeLensesWith abbreviatedFields ''SwarmGuide


-- | A bee in our swarm.
--
-- * @p@ is the type of a vector in our vector space \(\mathcal{H}\), i.e. type
--   of bee's position and velocity.
-- * @r@ is the type of an element of \((R,\leq)\), i.e. what the function we're
--   trying to minimise returns.
--
-- There are 'pos', 'vel', and 'guide' lenses to access position, velocity and
-- \"memory\" respectively.
data Bee p r g = Bee
  { _beePos :: !p -- ^ \(\in\mathcal{V}\), current position of the bee.
  , _beeVel :: !p -- ^ \(\in\mathcal{H}\), current velocity of the bee.
  , _beeVal :: !r -- ^ sdfkhsadf sdfkhsadf
  , _beeGuide :: !g -- ^ \"Memory\" of the bee.
  } deriving (Show, Eq)

makeLensesWith abbreviatedFields ''Bee


class (Ord r, HasPos g p, HasVal g r) => LocalGuide g p r where
  mkLocalG :: p -> p -> r -> g
  upLocalG :: Bee p r g -> Bee p r g

class (LocalGuide g p r) => GlobalGuide s g p r where
  mkGlobalG :: [Bee p r g] -> s


instance (Ord r) => LocalGuide (BeeGuide p r) p r where
  mkLocalG p _ fp = BeeGuide p fp
  upLocalG x
    | x ^. val < x ^. guide . val = x & (guide .~ (BeeGuide (x ^. pos)
                                                            (x ^. val)))
    | otherwise = x

instance (Num p, Ord r) => GlobalGuide (BeeGuide p r) (BeeGuide p r) p r where
  mkGlobalG xs = bestGuide xs

instance (Ord r, Fractional r, Foreign.Storable r)
  => GlobalGuide (SwarmGuide p r) (BeeGuide p r) p r where
    mkGlobalG xs =
      let g = bestGuide xs
          var = variance . V.fromList . map (view val) $ xs
       in SwarmGuide (g ^. pos) (g ^. val) var


mean :: (Fractional a, GV.Vector v a) => v a -> a
mean xs = GV.sum xs / fromIntegral (GV.length xs)

variance :: (Fractional a, GV.Vector v a) => v a -> a
variance xs = let avg = mean xs
                  n = fromIntegral . GV.length $ xs
               in (GV.sum . GV.map (\x -> (x - avg)^^2) $ xs) / n


bestGuide :: (LocalGuide g p r) => [Bee p r g] -> g
bestGuide = List.minimumBy (compBy (view val)) . map (view guide)
  where
    compBy f x y = compare (f x) (f y)

-- | Holds the \(\operatorname{updater}\) function from the discussion above.
-- Given a bee to update, global guide, and iteration, returns the updated
-- velocity of the bee. Well, almost. It actually returns a \"computation\"
-- that, given a random number generator of type @g@, can be run in monad @m@ to
-- produce the updated velocity as result. The reason why we return
-- @'ReaderT' g m p@ rather than just @p@ is randomness. Bees don't just move in
-- straight lines towards the commonly best known position, their movement is
-- stochastic with a bias towards moving in that direction. Hence, we need at
-- least @'Reader' g@. But then the fastest PRNG libraries use mutable state.
-- We thus need a 'PrimMonad' ('IO' or 'ST') where to run the computation.
-- Hence, 'ReaderT'.
newtype Updater m gen s g p r = Updater
  { runUpdater :: s -> Bee p r g -> ReaderT gen m p
  }

data Swarm m gen s g p r = Swarm
  { _swarmBees :: ![Bee p r g]
  , _swarmStats :: !s
  , _swarmFunc :: !(p -> ReaderT gen m r)
  , _swarmBounds :: !(p -> Bool)
  , _swarmUpdater :: !(Updater m gen s g p r)
  , _swarmIteration :: !Int
  }

makeLensesWith abbreviatedFields ''Swarm

-- | Updaters can be combined to build more difficult ones.
--
-- * 'mempty' creates a dummy updater, it just returns the velocity of the bee.
-- * 'mappend' ('<>') is almost like function composition, except that we read
-- it from left to right.  So @up1 <> up2@ means apply @up1@, update velocity of
-- the bee and then apply @up2@ to the \"updated\" bee.
instance (Monad m) => Monoid (Updater m gen s g p r) where
  mempty = Updater $ \_ x -> return (x ^. vel)
  mappend a b =
    let update stats bee = do
          v' <- runUpdater a stats bee
          runUpdater b stats (bee & vel .~ v')
     in Updater update

-- | Creates a local updater, i.e. one that changes velocity using local guide.
--
-- Suppose we have a bee at \(x\in\mathcal{V}\subseteq\mathcal{H}\) moving with
-- velocity \(v\in\mathcal{H}\) where (\mathcal{H}\) is a vector space over
-- field \(\mathbb{F}\). Suppose also that local best is given by
-- \(p\in\mathcal{V}\). Then, given a constant \(\varphi_l\in\mathbb{F}\), the
-- new velocity is
-- \[
--    v + \varphi_l U(0, 1) \circ (p - x) \;,
-- \]
-- where \(\circ\) denotes the Hadamard product and \(U(0, 1)\) is a vector
-- whose elements are uniformly distributed random in the \([0, 1)\) interval.
upLocal :: (RandomScalable p gen m, VectorSpace p a, HasPos g p)
        => a -> Updater m gen s g p r
upLocal c = Updater $ upLocalImpl c

upLocalImpl ::
     (RandomScalable p gen m, VectorSpace p a, HasPos g p)
  => a -- ^ Constant factor \(\phi_l\)
  -> s
  -> Bee p r g -- ^ Bee to update
  -> ReaderT gen m p
upLocalImpl c _ x = do
  randGen <- ask
  v' <- lift $ randScale ((x ^. guide . pos) - (x ^. pos)) randGen
  return $ x ^. vel + c `scale` v'


upLocalSimple :: (Randomisable a gen m, VectorSpace p a, HasPos g p)
  => a -> Updater m gen s g p r
upLocalSimple c = Updater $ f c
  where f c _ x = do
                    randGen <- ask
                    r <- lift $ random randGen
                    return $ x ^. vel + (c * r) `scale` ((x ^. guide . pos) - (x ^. pos))

-- | Creates a global updater, i.e. one that changes velocity using global
-- guide.
--
-- Suppose we have a bee at \(x\in\mathcal{V}\subseteq\mathcal{H}\) moving with
-- velocity \(v\in\mathcal{H}\) where \(\mathcal{H}\) is a vector space over
-- field \(\mathbb{F}\). Suppose also that global best of the whole swarm is
-- given by \(g\in\mathcal{V}\). Then, given a constant
-- \(\varphi_g\in\mathbb{F}\), the new velocity is
-- \[
--    v + \varphi_g U(0, 1) \circ (p - x) \;,
-- \]
-- where \(\circ\) denotes the Hadamard product and \(U(0, 1)\) is a vector
-- whose elements are uniformly distributed random in the \([0, 1)\) interval.
upGlobal :: (RandomScalable p gen m, VectorSpace p a, HasPos s p)
         => a -> Updater m gen s g p r
upGlobal c = Updater $ upGlobalImpl c

upGlobalImpl ::
     (RandomScalable p gen m, VectorSpace p a, HasPos s p)
  => a -- ^ Constant factor \(\phi_g\)
  -> s
  -> Bee p r g -- ^ Bee to update
  -> ReaderT gen m p
upGlobalImpl c stats x = do
  randGen <- ask
  v' <- lift $ randScale (stats ^. pos - x ^. pos) randGen
  return $ x ^. vel + c `scale` v'

upGlobalSimple :: (Randomisable a gen m, VectorSpace p a, HasPos s p)
  => a -> Updater m gen s g p r
upGlobalSimple c = Updater $ f c
  where f c stats x = do
                        randGen <- ask
                        r <- lift $ random randGen
                        return $ x ^. vel + (c * r) `scale` (stats ^. pos - x ^. pos)

-- | Creates a scaling updater, i.e. one that just scales the velocity.
--
-- Suppose we have a bee moving with velocity \(v\in\mathcal{H}\) where
-- \(\mathcal{H}\) is a vector space over field \(\mathbb{F}\). Then, given a
-- constant \(\omega\in\mathbb{F}\), the new velocity is \(\omega v\).
upScale :: (Monad m, Scalable a p) => a -> Updater m gen s g p r
upScale c = upScaleDyn $ const c

upScaleDyn :: (Monad m, Scalable a p) => (s -> a) -> Updater m gen s g p r
upScaleDyn c = Updater $ upScaleImpl c

upScaleImpl ::
     (Monad m, Scalable a p)
  => (s -> a)
  -> s
  -> Bee p r g
  -> ReaderT gen m p
upScaleImpl c stats x = return $ (c stats) `scale` (x ^. vel)

-- | Creates the standard \"WPG\" updater.
--
-- Suppose we have a bee at \(x\in\mathcal{V}\subseteq\mathcal{H}\) moving with
-- velocity \(v\in\mathcal{H}\) (\(\mathcal{H}\) is a vector space over field
-- \(\mathbb{F}\)). Suppose also that local best is \(p\in\mathcal{V}\) and
-- global best of the whole swarm is \(g\in\mathcal{V}\). Then, given a triple
-- \((\omega,\varphi_l,\varphi_g)\in\mathbb{F}^3\), the new
-- velocity is
-- \[
--    \omega v + \varphi_l U_l(0, 1) \circ (p - x)
--             + \varphi_g U_g(0, 1) \circ (p - x) \;,
-- \]
-- where \(\circ\) denotes the Hadamard product and \(U_l(0, 1)\) and
-- \(U_g(0,1)\) are vectors whose elements are uniformly distributed random in
-- the \([0, 1)\) interval.
standardUpdater ::
     (RandomScalable p gen m, VectorSpace p a, HasPos g p, HasPos s p)
  => (a, a, a) -> Updater m gen s g p r
standardUpdater (w, cl, cg) = (upScale w) <> (upLocal cl) <> (upGlobal cg)

standardUpdaterSimple ::
     (Randomisable a gen m, VectorSpace p a, HasPos g p, HasPos s p)
  => (a, a, a) -> Updater m gen s g p r
standardUpdaterSimple (w, cl, cg) = (upScale w)
                            <> (upLocalSimple cl)
                            <> (upGlobalSimple cg)


randomWalkUpdater :: (UniformDist p gen m)
                  => (p, p) -> Updater m gen s g p r
randomWalkUpdater bounds = Updater $ f
  where f _ _ = do
          randGen <- ask
          v' <- lift $ uniform bounds randGen
          return v'

kickUpdater ::
     ( RandomScalable p gen m
     , UniformDist p gen m
     , VectorSpace p a
     , Scalable r p
     , HasVar s r
     , HasPos s p
     , HasPos g p
     , Floating r
     , Ord r
     , s ~ SwarmGuide p r
     )
  => (r, r) -> (p, p) -> (a, a, a) -> Updater m gen s g p r
kickUpdater cutoff bounds wpg = Updater $ kickUpdaterImpl cutoff bounds wpg

kickUpdaterImpl ::
     ( RandomScalable p gen m
     , UniformDist p gen m
     , VectorSpace p a
     , Scalable r p
     , HasVar s r
     , HasPos s p
     , HasPos g p
     , Floating r
     , Ord r
     )
  => (r, r) -> (p, p) -> (a, a, a) -> s -> Bee p r g -> ReaderT gen m p
kickUpdaterImpl (cutoff, c) bounds wpg stats x = runUpdater updater stats x
  where updater = if (stats ^. var) < cutoff
                     then randomWalkUpdater bounds
                          <> upScale (c * sqrt (stats ^.var))
                     else standardUpdater wpg


-- | Given a swarm and bee in the @n@'th iteration, returns a computation that
-- returns the bee in the @(n+1)@'st iteration.
updateBee ::
     (Monad m, Num p, Ord r, LocalGuide g p r)
  => Swarm m gen s g p r -- ^ Swarm in @n@'th iteration.
  -> Bee p r g  -- ^ Bee in @n@'th iteration.
  -> ReaderT gen m (Bee p r g) -- ^ Bee in @(n+1)@'st iteration.
updateBee xs x = do
  v <- runUpdater (xs ^. updater) (xs ^. stats) x
  let p = x ^. pos + v
  if (xs ^. bounds) p
     then do fp <- (xs ^. func) p
             return $ upLocalG $ x & (pos .~ p) & (vel .~ v) & (val .~ fp)
     else return $ x & (vel .~ (-v))


-- | Given a swarm in the @n@'th iteration, returns a computation that returns
-- the swarm in the @(n+1)@'st iteration.
updateSwarm :: (Monad m, Num p, Ord r, GlobalGuide s g p r)
            => Swarm m gen s g p r -> ReaderT gen m (Swarm m gen s g p r)
updateSwarm swarm = do
  xs <- mapM (updateBee swarm) (swarm ^. bees)
  return $ swarm & (bees .~ xs)
                 & (stats .~ (mkGlobalG xs))
                 & (iteration +~ 1)

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
     (Monad m, GlobalGuide s g p r)
  => Updater m gen s g p r -- ^ Update strategy.
  -> ((p -> ReaderT gen m r) -> ReaderT gen m (Bee p r g)) -- ^ Bee initialiser.
  -> (p -> Bool) -- ^ Boundaries
  -> (p -> ReaderT gen m r) -- ^ Function to minimise
  -> Int -- ^ Number of bees in the swarm.
  -> ReaderT gen m (Swarm m gen s g p r)
mkSwarm updater initialiser boundaries func n = do
  xs <- sequence . replicate n $ initialiser func
  return $ Swarm xs (mkGlobalG xs) func boundaries updater 0

fromPure :: (Monad m) => (p -> r) -> p -> ReaderT gen m r
fromPure f x = reader $ \_ -> f x

class Between a b where
  isBetween :: (a, a) -> b -> Bool

instance (Ord a) => Between a a where
  isBetween (l, h) x = l <= x && x <= h

instance {-# OVERLAPS #-} (Between a a) => Between (Complex a) (Complex a) where
  isBetween (rl :+ il, rh :+ ih) (rx :+ ix) =
    isBetween (rl, rh) rx && isBetween (il, ih) ix

instance {-# OVERLAPS #-} (Foreign.Storable a, Between a a)
  => Between (V.Vector a) (V.Vector a) where
    isBetween (low, high) x =
      V.and $ V.zipWith3 (\l h x -> isBetween (l, h) x) low high x

-- | Creates a new 'Bee' with a random position inside a given 1D interval and
-- zero velocity.
randomBee1D ::
     (Num p, UniformDist p gen m)
  => (p, p) -- ^ @[low, high]@ interval. It is assumed that @low <= high@.
  -> (p -> ReaderT gen m r) -- ^ Function we're minimising.
  -> ReaderT gen m (Bee p r (BeeGuide p r))
randomBee1D bounds func = do
  g <- ask
  x <- lift $ uniform bounds g
  fx <- func x
  return $ Bee x 0 fx (BeeGuide x fx)

-- | Creates a new 'Bee' with a random position inside a given N-dimensional
-- interval and zero velocity.
randomBeeND ::
     ( Foreign.Storable a
     , Num a
     , UniformDist p gen m
     , p ~ V.Vector a
     , LocalGuide g p r)
  => (V.Vector a, V.Vector a) -- ^ \(=(a, b)\) with \(a,b\in\mathcal{H}\). It
  -- is assumed that
  -- \(\forall i\in\{0,\dots,\dim(\mathcal{H})-1\}: a_i\leq b_i\). \"Valid\"
  -- ND interval is then defined as
  -- \(\mathcal{V}=\prod_{i=0}^{\dim(\mathcal{H})-1} [a_i, b_i]\).
  -> (V.Vector a -> ReaderT gen m r) -- ^ Function \(f\) we're minimising.
  -> ReaderT gen m (Bee p r g)
randomBeeND bounds func = do
  randGen <- ask
  x <- lift $ uniform bounds randGen
  let v = V.replicate (V.length x) 0
  fx <- func x
  return $ Bee x v fx $ mkLocalG x v fx

-- | Simplified @mkSwarm@ for 1D pure functions.
-- simpleInit1D ::
--      ( Num a
--      , RandomScalable a gen m
--      , UniformDist a gen m
--      , Between a a
--      , Ord r
--      , s ~ g
--      , g ~ BeeGuide a r
--      )
--   => (a, a, a) -- ^ WPG parameters \((\omega,\varphi_l,\varphi_g)\).
--   -> (a, a) -- ^ Boundaries \([a, b]\).
--   -> (a -> r) -- ^ Function \(f\) we're minimising.
--   -> Int -- ^ Number of bees.
--   -> ReaderT gen m (Swarm m gen s g a r)
-- simpleInit1D wpg (low, high) func =
--   let updater = standardUpdater wpg
--       initialiser = randomBee1D (low, high)
--       boundaries = isBetween (low, high)
--   in mkSwarm updater initialiser boundaries (fromPure func)


unpackToV :: (Foreign.Storable a) => [(a, a)] -> (V.Vector a, V.Vector a)
unpackToV xs = let a = V.fromList . map fst $ xs
                   b = V.fromList . map snd $ xs
                in (a, b)

-- | Simplified @mkSwarm@ for N-dimensional functions.
simpleInitND ::
     ( Foreign.Storable a
     , VectorSpace p a
     , RandomScalable p gen m
     , UniformDist p gen m
     , Between p p
     , Ord r
     , p ~ V.Vector a
     , GlobalGuide s g p r
     )
  => Updater m gen s g p r -- ^ PWG parameters \((\omega,\varphi_l,\varphi_g)\)
  -> [(a, a)]
  -> [(a, a)] -- ^ List of intervals \([a_i,b_i]\). \(\mathcal{V}\) is then
  -- \(\prod_i [a_i,b_i]\).
  -> (p -> r) -- ^ Function \(f\) we're minimising.
  -> Int -- ^ Number of bees.
  -> ReaderT gen m (Swarm m gen s g p r)
simpleInitND updater initBounds bounds func =
  let initialiser = randomBeeND (unpackToV initBounds)
      boundaries = isBetween (unpackToV bounds)
  in mkSwarm updater initialiser boundaries (fromPure func)

-- optimise1D ::
--      ( Num a
--      , RandomScalable a gen m
--      , UniformDist a gen m
--      , Between a a
--      , Ord r
--      , s ~ g
--      , g ~ BeeGuide a r
--      )
--   => (a, a, a)
--   -> (a, a)
--   -> (a -> r)
--   -> Int
--   -> (Swarm m gen s g a r -> Bool)
--   -> ReaderT gen m [Swarm m gen s g a r]
-- optimise1D wpg bounds func n predicate = do
--   swarm <- simpleInit1D wpg bounds func n
--   iterateWhileM (not . predicate) updateSwarm swarm

optimiseND ::
     ( Foreign.Storable a
     , VectorSpace p a
     , RandomScalable p gen m
     , UniformDist p gen m
     , Between p p
     , Ord r
     , p ~ V.Vector a
     , GlobalGuide s g p r
     )
  => Updater m gen s g p r
  -> [(a, a)]
  -> [(a, a)]
  -> (p -> r)
  -> Int
  -> (Swarm m gen s g p r -> Bool)
  -> ReaderT gen m [Swarm m gen s g p r]
optimiseND wpg initBounds bounds func n predicate = do
  swarm <- simpleInitND wpg initBounds bounds func n
  iterateWhileM (not . predicate) updateSwarm swarm


