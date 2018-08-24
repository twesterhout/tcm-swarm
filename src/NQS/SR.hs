{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE CPP #-}


module NQS.SR
  ( -- * Stochastic Reconfiguration
    --
    -- Globally, the SR algorithm looks more or less like this
    --
    -- @
    --     for n in {0,...,maxIter - 1} do
    --         (F, ∂ψ) <- sample ψ
    --         S <- covariance⁽ⁿ⁾ ∂ψ
    --         δ <- (S + λ⁽ⁿ⁾)⁻¹ F
    --         ψ <- ψ - learningRate * δ
    --     done
    -- @

    -- * AXPY
    -- * Sampling
    -- * Constructing S
    -- * Solving S
    sr
  , IterInfo(..)
  , HasIter(..)
  , HasSampler(..)
  , HasSolver(..)
  , HasMoments(..)
  , HasForceNorm(..)
  ) where

import           Prelude                 hiding ( zipWith
                                                , zipWithM
                                                , map
                                                , mapM
                                                )

import           GHC.Generics                   ( Generic )

import           Debug.Trace
import           Control.Exception              ( assert )
import           Control.Monad.Identity         ( Identity(..) )
import           Control.Monad.ST
import           System.IO.Unsafe               ( unsafePerformIO )
import           Foreign.Storable
import           Foreign.ForeignPtr
import           Data.Vector.Storable           ( Vector )
import qualified Data.Vector.Storable          as V
import           Data.Vector.Storable.Mutable   ( MVector )
import qualified Data.Vector.Storable.Mutable  as MV
import qualified Data.Vector.Unboxed

import           Data.Singletons
import           Data.Complex
import           Data.Semigroup                 ( (<>) )
import           Control.Monad                  ( (>=>) )
import           Control.Monad.Primitive

import           Control.DeepSeq
import           System.CPUTime
import Data.Aeson

import qualified NQS.CG                        as CG
import           NQS.CG                         ( Operator )
import           NQS.Rbm (Rbm(..))
import           NQS.Rbm.Mutable
import           NQS.Internal.BLAS
import           NQS.Internal.LAPACK
import           NQS.Internal.Types
import           NQS.Internal.Hamiltonian
import           NQS.Internal.Sampling
import           NQS.Internal.Rbm (unsafeFreezeRbm)

import           Lens.Micro
import           Lens.Micro.TH
import           Lens.Micro.Extras

import           GHC.Float                      ( int2Float )


import           Data.Aeson
import qualified Data.ByteString.Lazy.Char8     as BS

-- | In Stochastic Reconfiguration we only ever deal with 'Direct' vectors.
type V = MDenseVector 'Direct

-- | A shorter name for dense matrices.
type M orient s a = MDenseMatrix orient s a


data SolverStats = SolverStats
  { _solverStatsIters :: {-# UNPACK #-}!Int
  , _solverStatsErr   :: {-# UNPACK #-}!ℝ
  , _solverStatsTime  :: {-# UNPACK #-}!Double
  }

makeFields ''SolverStats

instance ToJSON SolverStats where
  toJSON stats =
    object [ "iters" .= (stats ^. iters)
           , "error" .= (stats ^. err)
           , "time"  .= (stats ^. time)
           ]
  toEncoding stats =
    pairs $ "iters" .= (stats ^. iters)
         <> "error" .= (stats ^. err)
         <> "time"  .= (stats ^. time)

instance FromJSON SolverStats where
  parseJSON = withObject "SolverStats" $ \v ->
    SolverStats <$> v .: "iters"
                <*> v .: "error"
                <*> v .: "time"


data SamplerStats = SamplerStats
  { _samplerStatsMoments  :: {-# UNPACK #-}!(Vector ℂ)
  , _samplerStatsStdDev   :: !(Maybe ℝ)
  , _samplerStatsDim      :: {-# UNPACK #-}!Int
  , _samplerStatsTime     :: {-# UNPACK #-}!Double
  }

makeFields ''SamplerStats

instance ToJSON SamplerStats where
  toJSON stats =
    object [ "moments" .= (stats ^. moments)
           , "stddev"  .= (stats ^. stdDev)
           , "dim"     .= (stats ^. dim)
           , "time"    .= (stats ^. time)
           ]
  toEncoding stats =
    pairs $ "moments" .= (stats ^. moments)
         <> "stddev"  .= (stats ^. stdDev)
         <> "dim"     .= (stats ^. dim)
         <> "time"    .= (stats ^. time)

instance FromJSON SamplerStats where
  parseJSON = withObject "SamplerStats" $ \v ->
    SamplerStats <$> v .: "moments"
                 <*> v .: "stddev"
                 <*> v .: "dim"
                 <*> v .: "time"

data IterInfo = IterInfo
  { _iterInfoIter      :: {-# UNPACK #-}!Int
  , _iterInfoState     :: Rbm
  , _iterInfoSampler   :: !SamplerStats
  , _iterInfoSolver    :: !SolverStats
  , _iterInfoForceNorm :: {-# UNPACK #-}!ℝ
  , _iterInfoDeltaNorm :: {-# UNPACK #-}!ℝ
  }

makeFields ''IterInfo

instance ToJSON IterInfo where
  toJSON stats =
    object [ "iter"      .= (stats ^. iter)
           , "state"     .= (stats ^. state)
           , "sampler"   .= (stats ^. sampler)
           , "solver"    .= (stats ^. solver)
           , "forceNorm" .= (stats ^. forceNorm)
           , "deltaNorm" .= (stats ^. deltaNorm)
           ]
  toEncoding stats =
    pairs $ "iter"      .= (stats ^. iter)
         <> "state"     .= (stats ^. state)
         <> "sampler"   .= (stats ^. sampler)
         <> "solver"    .= (stats ^. solver)
         <> "forceNorm" .= (stats ^. forceNorm)
         <> "deltaNorm" .= (stats ^. deltaNorm)

instance FromJSON IterInfo where
  parseJSON = withObject "IterInfo" $ \v ->
    IterInfo <$> v .: "iter"
             <*> v .: "state"
             <*> v .: "sampler"
             <*> v .: "solver"
             <*> v .: "forceNorm"
             <*> v .: "deltaNorm"

-- AXPY
-- -----------------------------------------------------------------------------

type AxpyFun v m = ℂ -> V (PrimState m) ℂ -> v (PrimState m) -> m ()

-- | \(Y \leftarrow \alpha X + Y\) for 'Column' matrices.
axpyMatrix
  :: (PrimMonad m, Storable a, AXPY (MDenseVector 'Direct) a)
  => a
  -> MDenseVector 'Direct (PrimState m) a
  -> MDenseMatrix 'Column (PrimState m) a
  -> m ()
axpyMatrix !α !x !y@(MDenseMatrix !rows !cols !stride !buff) =
  assert (rows * cols == x ^. dim) $ loop 0
 where
  getX !i = slice (i * rows) rows x
  getY !i = unsafeColumn i y
  loop !i | i < cols  = axpy α (getX i) (getY i) >> loop (i + 1)
          | otherwise = return ()

-- | Performs @y <- αx + y@ where @x@ is a vector and @y@ -- an RBM.
axpyRbm :: AxpyFun MRbm IO
axpyRbm α x y =
  let n = sizeVisible y
      m = sizeHidden y
  in  assert (x ^. dim == size y) $ do
        withVisible y $ \a -> do
          axpy α (slice 0 n x) a
          -- unsafeFreeze a >>= \a' -> print (a' ^. buffer)
        withHidden y $ \b -> do
          axpy α (slice n m x) b
          -- unsafeFreeze b >>= \b' -> print (b' ^. buffer)
        withWeights y $ \w -> do
          axpyMatrix α (slice (n + m) (n * m) x) w
          -- unsafeFreeze w >>= \w' -> print (w' ^. buffer)

-- Monte-Carlo sampling
-- -----------------------------------------------------------------------------

-- | Sampling
type SampleFun v m
  = v (PrimState m)
 -> V (PrimState m) ℂ
 -> M 'Row (PrimState m) ℂ
 -> m SamplerStats

-- | Monte-Carlo sampling an RBM
sampleRbm :: MCConfig -> Hamiltonian -> SampleFun MRbm IO
sampleRbm config hamiltonian = doSample
 where
  doSample rbm force derivatives = do
    t1                    <- getCPUTime
    moments               <- MV.new 2
    (dimension, variance) <- sampleGradients config
                                             hamiltonian
                                             rbm
                                             moments
                                             force
                                             derivatives
    t2 <- getCPUTime
    V.unsafeFreeze moments >>= \moments' -> return
      (SamplerStats moments'
                    (sqrt <$> variance)
                    dimension
                    (fromIntegral (t2 - t1) * 1.0E-12)
      )

type SolveFun wrapper m
  = wrapper -- ^ S
 -> V (PrimState m) ℂ -- ^ b
 -> V (PrimState m) ℂ -- ^ x
 -> m SolverStats

type MakeSFun wrapper m = Int -> M 'Row (PrimState m) ℂ -> m wrapper

-- | Sparse representation of the @S@ matrix.
data SMatrix s =
  SMatrix !(M 'Row s ℂ) -- ^ Derivatives (O - 〈O〉)
          !(V s ℂ) -- ^ Workspace of size #steps.
          !(Maybe ℂ) -- ^ Regulariser λ

makeS :: PrimMonad m => Maybe (Int -> ℂ) -> MakeSFun (SMatrix (PrimState m)) m
makeS regulariser i gradients = do
  workspace <- newDenseVector (gradients ^. dim ^. _1)
  return $! SMatrix gradients workspace ((\f -> f i) <$> regulariser)

makeS' :: PrimMonad m => Maybe (Int -> ℂ) -> MakeSFun (M 'Row (PrimState m) ℂ) m
makeS' regulariser i gradients = do
  let (steps, params) = gradients ^. dim
  s <- newDenseMatrix params params
  herk MatUpper ConjTranspose (1 / int2Float steps) gradients 0 s
  herk MatLower ConjTranspose (1 / int2Float steps) gradients 0 s
  case regulariser of
    Just f  -> do
      one <- MDenseVector @'Direct params 0 <$> V.unsafeThaw (V.singleton 1)
      axpy (f i) one (MDenseVector @'Direct params (s ^. stride + 1) (s ^. buffer))
    Nothing -> return ()
  return $! s

opS :: PrimMonad m => SMatrix (PrimState m) -> Operator m ℂ
opS (SMatrix o temp λ) x out = do
  let scale = (1 / int2Float (o ^. dim ^. _1)) :+ 0
  gemv NoTranspose 1 o x 0 temp
  case λ of
    (Just λ') -> copy x out >> gemv ConjTranspose scale o temp λ' out
    Nothing   -> gemv ConjTranspose scale o temp 0 out

{-
solveS :: PrimMonad m => CGConfig ℝ -> SolveFun (MDenseMatrix 'Row (PrimState m) ℂ) m
solveS (CGConfig maxIter tol) = doSolve
  where doSolve s b x = do
          let operator = \input output -> gemv NoTranspose 1 s input 0 output
          zero <- MDenseVector @'Direct (b ^. dim) 0 <$> V.unsafeThaw (V.singleton 0)
          copy zero x
          !answer <- CG.cg maxIter tol operator b x
          trace (show answer) (return ())
-}

solveS :: CGConfig ℝ -> SolveFun (SMatrix RealWorld) IO
solveS (CGConfig maxIter tol) = doSolve
 where
  doSolve s b x = do
    t1  <- getCPUTime
    one <- MDenseVector @ 'Direct (b ^. dim) 0 <$> V.unsafeThaw (V.singleton 0)
    copy one x
    (iters, err) <- CG.cg maxIter tol (opS s) b x
    t2           <- getCPUTime
    return $! SolverStats iters err (fromIntegral (t2 - t1) * 1.0E-12)

solveS' :: SolveFun (MDenseMatrix 'Row RealWorld ℂ) IO
solveS' s b x = assert (x ^. stride == 1) $ do
  t1 <- getCPUTime
  copy b x
  cgelsd s (MDenseMatrix (b ^. dim) 1 1 (x ^. buffer)) (-1.0)
  t2 <- getCPUTime
  return $! SolverStats (-1) (-1.0) (fromIntegral (t2 - t1) * 1.0E-12)


type Stepper v m a = v (PrimState m) a -> m ()


numberSteps :: (Int, Int, Int) -> Int
numberSteps (low, high, step) = (high - low - 1) `div` step + 1

newWorkspace :: PrimMonad m => Int -> Int -> m (DenseWorkspace (PrimState m) ℂ)
newWorkspace nParams nSteps =
  DenseWorkspace
    <$> newDenseVector nParams
    <*> newDenseMatrix nSteps nParams
    <*> newDenseVector nParams

sr
  :: SRConfig ℂ
  -> Hamiltonian
  -> MRbm (PrimState IO)
  -> (IterInfo -> IO ())
  -> IO ()
sr config hamiltonian ψ process = newWorkspace nParams nSteps
  >>= \w -> loop w 0
 where
  nParams  = size ψ
  nSteps   = config ^. mc ^. runs * numberSteps (config ^. mc ^. steps)
  doAxpy   = axpyRbm
  doMakeS  = makeS (config ^. regulariser)
  doSolveS = solveS (config ^. cg)
  doSample = sampleRbm (config ^. mc) hamiltonian
  loop !w@(DenseWorkspace f grad δ) !i
    | i >= config ^. maxIter = return ()
    | otherwise = do
      mcStats <- doSample ψ f grad
      fNorm   <- nrm2 f
      cgStats <- doMakeS i grad >>= \s -> doSolveS s f δ
      δNorm   <- nrm2 δ
      unsafeFreezeRbm ψ
        >>= \ψ' ->
              let stats = IterInfo i ψ' mcStats cgStats fNorm δNorm
              in  process stats
      doAxpy (-(config ^. rate) i) δ ψ
      loop w (i + 1)


