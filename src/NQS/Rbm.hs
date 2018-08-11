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
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP #-}

module NQS.Rbm
  ( Rbm(..)
  , mkRbm
  , IsRbm(..)
  , sampleMoments
  , Hamiltonian(..)
  , map
  , mapM
  , zipWith
  , zipWithM
  ) where

import Prelude hiding (zipWith, zipWithM, map, mapM)

import GHC.Generics (Generic)

import Debug.Trace
import Control.Exception (assert)
import Control.Monad.Identity (Identity(..))
import Control.Monad.ST
import System.IO.Unsafe (unsafePerformIO)
import Foreign.Storable
import Foreign.ForeignPtr
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable.Mutable (MVector)
import qualified Data.Vector.Storable.Mutable as MV
import qualified Data.Vector.Unboxed

import Data.Complex
import Data.Semigroup((<>))
import Control.Monad((>=>))
import Control.Monad.Primitive

import Control.DeepSeq

import NQS.Internal.Rbm
import NQS.Internal.BLAS
import NQS.Internal.Types

import Lens.Micro
import Lens.Micro.Extras

#if 0
type UVector = Data.Vector.Unboxed.Vector


data DenseWorkspace s a = DenseWorkspace
  { _denseWorkspaceDerivatives :: MDenseMatrix 'Row s a    -- ^ Derivatives
  , _denseWorkspaceForce :: MDenseVector 'Direct s a -- ^ Force
  , _denseWorkspaceCovariance :: MDenseMatrix 'Row s a -- ^ Covariance matrix S
  }

type SampleFun v m a = v a
                    -> MDenseVector 'Direct (PrimState m) a
                    -> MDenseMatrix 'Row (PrimState m) a
                    -> m (Estimate NormalError a)


class Storable a => MSRState (v :: * -> * -> *) a where
  axpyState :: PrimMonad m => a -> DenseVector 'Direct a -> v (PrimState m) a -> m ()

class MSRState (Mutable v) a => SRState (v :: * -> *) a where
  sampleGradients :: PrimMonad m
    => Int -> Hamiltonian -> (Int, Int, Int) -> Maybe Int -> SampleFun v m a

solveDense ::
     PrimMonad m
  => DenseMatrix ornt a
  -> DenseVector direct a
  -> MDenseVector direct (PrimState m) a
  -> m ()
solveDense = undefined
#endif

mkRbm :: IsRbm a => (Vector a, Vector a, Vector a) -> Rbm a
mkRbm (a, b, w)
  | V.length a * V.length b /= V.length w = error $!
    "createRbm: Incompatible dimensions: " <> show (V.length a, V.length b, V.length w)
  | otherwise = let n = V.length a
                    m = V.length b
                 in runST $ do rbm <- newRbm n m
                               withVisibleMut rbm (\x -> V.copy x a)
                               withHiddenMut rbm (\x -> V.copy x b)
                               withWeightsMut rbm (\x -> V.copy x w)
                               unsafeFreeze rbm

class Storable a => IsRbm a where
  size :: Rbm a -> Int
  sizeMut :: MRbm s a -> Int
  sizeVisible :: Rbm a -> Int
  sizeVisibleMut :: MRbm s a -> Int
  sizeHidden :: Rbm a -> Int
  sizeHiddenMut :: MRbm s a -> Int
  newRbm :: PrimMonad m => Int -> Int -> m (MRbm (PrimState m) a)
  withVisible :: Monad m => Rbm a -> (Vector a -> m b) -> m b
  withVisibleMut :: PrimMonad m => MRbm (PrimState m) a -> (MVector (PrimState m) a -> m b) -> m b
  withHidden :: Monad m => Rbm a -> (Vector a -> m b) -> m b
  withHiddenMut :: PrimMonad m => MRbm (PrimState m) a -> (MVector (PrimState m) a -> m b) -> m b
  withWeights :: Monad m => Rbm a -> (Vector a -> m b) -> m b
  withWeightsMut :: PrimMonad m => MRbm (PrimState m) a -> (MVector (PrimState m) a -> m b) -> m b

#if 0
instance (AXPY a, IsRbm a) => MSRState (Mutable Rbm) a where
  axpyState a x y = do
      let n = sizeVisibleMut y
          m = sizeHiddenMut y
      withVisible y (\v -> axpy a (slice 0 n) (fromTuple (n, 1, v)))
      withHidden y (\v -> axpy a (slice n m) (fromTuple (m, 1, v)))
      withWeights y (\v -> axpy a (slice (n + m) (n * m)) (fromTuple ((m * n), 1, v)))
    where !xStride = x ^. stride
          !xBuffer = x ^. buffer
          slice !i !size = fromTuple $!
            (size, xStride, V.slice (i * xStride) size xBuffer)
#endif

instance IsRbm (Complex Float) where
  size x = withRbmPure x _RbmC'size
  sizeMut x = withMRbmPure x _RbmC'size
  sizeVisible x = withRbmPure x _RbmC'sizeVisible
  sizeVisibleMut x = withMRbmPure x _RbmC'sizeVisible
  sizeHidden x = withRbmPure x _RbmC'sizeHidden
  sizeHiddenMut x = withMRbmPure x _RbmC'sizeHidden
  newRbm n m = MRbm <$> unsafeIOToPrim (_RbmC'new n m)
  withVisibleMut (MRbm x) f = withForeignPtrPrim x $ _RbmC'getVisible >=> f
  withHiddenMut (MRbm x) f = withForeignPtrPrim x $ _RbmC'getHidden >=> f
  withWeightsMut (MRbm x) f = withForeignPtrPrim x $ _RbmC'getWeights >=> f
  withVisible (Rbm x) f = unsafePerformIO $ withForeignPtr x $
    _RbmC'getVisible >=> V.unsafeFreeze >=> (return . f)
  withHidden (Rbm x) f = unsafePerformIO $ withForeignPtr x $
    _RbmC'getHidden >=> V.unsafeFreeze >=> (return . f)
  withWeights (Rbm x) f = unsafePerformIO $ withForeignPtr x $
    _RbmC'getWeights >=> V.unsafeFreeze >=> (return . f)

#if 0
instance SRState Rbm (Complex Float) where
  sampleGradients nRuns h steps m = \rbm mforce mderivatives ->
    withRbm rbm $ \rbmPtr ->
      _RbmC'sample rbmPtr h nRuns steps m mforce mderivatives
#endif


zipWithM ::
     forall a b c m. (IsRbm a, IsRbm b, IsRbm c, Monad m)
  => (a -> b -> m c) -> Rbm a -> Rbm b -> m (Rbm c)
zipWithM func a b = do
  visible <- withVisible a $ \a' -> withVisible b $ \b' -> V.zipWithM func a' b'
  hidden <- withHidden a $ \a' -> withHidden b $ \b' -> V.zipWithM func a' b'
  weights <- withWeights a $ \a' -> withWeights b $ \b' -> V.zipWithM func a' b'
  return $ mkRbm (visible, hidden, weights)

zipWith ::
     forall a b c. (IsRbm a, IsRbm b, IsRbm c)
  => (a -> b -> c) -> Rbm a -> Rbm b -> Rbm c
zipWith func a b = runIdentity $ zipWithM (\a b -> return $ func a b) a b

mapM :: (Monad m, IsRbm a, IsRbm b) => (a -> m b) -> Rbm a -> m (Rbm b)
mapM func x = do
  visible <- withVisible x $ \x' -> V.mapM func x'
  hidden <- withHidden x $ \x' -> V.mapM func x'
  weights <- withWeights x $ \x' -> V.mapM func x'
  return $ mkRbm (visible, hidden, weights)

map :: (IsRbm a, IsRbm b) => (a -> b) -> Rbm a -> Rbm b
map func = runIdentity . mapM (return . func)


instance (IsRbm a, Num a) => Num (Rbm a) where
  (+) x y = zipWith (+) x y
  (-) x y = zipWith (-) x y
  (*) x y = zipWith (*) x y
  abs x = map abs x
  signum x = map signum x
  negate x = map negate x
  fromInteger = undefined

sampleMoments ::
     PrimMonad m
  => Rbm (Complex Float)
  -> Hamiltonian
  -> Int
  -> (Int, Int, Int)
  -> Maybe Int
  -> Int
  -> m (V.Vector (Complex Float))
sampleMoments (Rbm x) h nRuns nSteps m n =
  withForeignPtrPrim x $ \xPtr ->
    _RbmC'sampleMoments xPtr h nRuns nSteps m n
  

{-
instance IsRbm a => Scalable a (Rbm a) where
  scale λ = map (*λ)

instance (PrimMonad m, DeltaWell m Float Float)
  => DeltaWell m Float (Rbm Float) where
    upDeltaWell κ p x = zipWithM (upDeltaWell κ) p x
-}


{-
type Regularizer m a = Int -> MDenseMatrix 'Row (PrimState m) a -> m ()


getDiag :: Storable a => DenseMatrix orient a -> DenseVector 'Direct a
getDiag mat@(view dim -> (xdim, ydim))
  | xdim /= ydim = error $
  "getDiag: Cannot extract the diagonal of a non-diagonal matrix with \
  \dimensions (" <> show (xdim, ydim) <> "."
  | otherwise = toBLASVector $! (xdim, mat ^. stride + 1, mat ^. buffer)

getDiagM :: (PrimMonad m, Storable a)
         => MDenseMatrix orient (PrimState m) a
         -> m (MDenseVector 'Direct (PrimState m) a)
getDiagM mat@(view dim -> (xdim, ydim))
  | xdim /= ydim = error $
  "getDiagM: Cannot extract the diagonal of a non-diagonal matrix with \
  \dimensions (" <> show (xdim, ydim) <> "."
  | otherwise = return $! toBLASVector (xdim, mat ^. stride + 1, mat ^. buffer)

class MakeRegularizer a where
  makeRegularizer :: PrimMonad m => Maybe (Int -> a) -> Regularizer m a

instance MakeRegularizer Float where
  makeRegularizer (Just f) = \i -> getDiagM >=> \diag -> let !a = 1.0 + f i in scal a diag
  makeRegularizer Nothing = \i _ -> return ()

type MakeDenseS m a = Int
                   -> DenseMatrix 'Row a -- ^ Derivatives
                   -> MDenseMatrix 'Row (PrimState m) a -- ^ Covariance matrix
                   -> m ()

type DenseSolver m a = MDenseMatrix 'Row (PrimState m) a
                    -> MDenseVector 'Direct (PrimState m) a
                    -> m ()

type Stepper v m a = v (PrimState m) a -> m ()

stepSR ::
     (PrimMonad m, Storable a, Num a, SRState v a, FreezeThaw v a)
  => SampleFun v m a
  -> MakeDenseS m a
  -> DenseSolver m a
  -> DenseWorkspace (PrimState m) a
  -> Int
  -> (Mutable v) (PrimState m) a
  -> m (Estimate NormalError a)
stepSR sample makeS solveSF (DenseWorkspace mderivatives mforce mcovariance) i mx =
  do
    x <- unsafeFreeze mx
    energyEstimate <- sample x mforce mderivatives
    derivatives <- unsafeFreeze mderivatives
    makeS i derivatives mcovariance
    solveSF mcovariance mforce
    force <- unsafeFreeze mforce
    axpyState (-1) force mx
    return energyEstimate
-}



