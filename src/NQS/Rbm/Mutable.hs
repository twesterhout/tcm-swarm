{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}

module NQS.Rbm.Mutable
  ( MRbm(..)
  , new
  , size
  , sizeVisible
  , sizeHidden
  , sizeWeights
  , withVisible
  , withHidden
  , withWeights
  , setVisible
  , setHidden
  , setWeights
  , mapM
  , zipWithM
  , sampleMoments
  , sampleGradients
  ) where

import Prelude hiding (zipWith, zipWithM, map, mapM)

import Data.Singletons (SingI)
import Control.Monad.Identity (Identity(..))
import Control.Monad.Primitive

import NQS.Internal.BLAS
import NQS.Internal.Rbm (MRbm(..))
import NQS.Internal.Types
import NQS.Internal.Sampling
import qualified NQS.Internal.Rbm as Core


size :: MRbm s -> Int
size !(MRbm x) = Core.size x

sizeVisible :: MRbm s -> Int
sizeVisible !(MRbm x) = Core.sizeVisible x

sizeHidden :: MRbm s -> Int
sizeHidden !(MRbm x) = Core.sizeHidden x

sizeWeights :: MRbm s -> Int
sizeWeights !(MRbm x) = Core.sizeWeights x

withVisible ::
     PrimMonad m
  => MRbm (PrimState m)
  -> (MDenseVector 'Direct (PrimState m) ℂ -> m r)
  -> m r
withVisible !(MRbm x) f = Core.withVisible x f

withHidden ::
     PrimMonad m
  => MRbm (PrimState m)
  -> (MDenseVector 'Direct (PrimState m) ℂ -> m r)
  -> m r
withHidden !(MRbm x) f = Core.withHidden x f

withWeights ::
     PrimMonad m
  => MRbm (PrimState m)
  -> (MDenseMatrix 'Column (PrimState m) ℂ -> m r)
  -> m r
withWeights !(MRbm x) f = Core.withWeights x f

setVisible ::
     PrimMonad m
  => MRbm (PrimState m)
  -> MDenseVector 'Direct (PrimState m) ℂ
  -> m ()
setVisible rbm x = withVisible rbm $ copy x

setHidden ::
     PrimMonad m
  => MRbm (PrimState m)
  -> MDenseVector 'Direct (PrimState m) ℂ
  -> m ()
setHidden rbm x = withHidden rbm $ copy x

setWeights ::
     (PrimMonad m, SingI orient)
  => MRbm (PrimState m)
  -> MDenseMatrix orient (PrimState m) ℂ
  -> m ()
setWeights rbm x = withWeights rbm $ \w -> loop w 0
  where n = sizeVisible rbm
        loop w i
          | i < n = copy (unsafeColumn i x) (unsafeColumn i w) >> loop w (i + 1)
          | otherwise = return ()

new :: PrimMonad m => Int -> Int -> m (MRbm (PrimState m))
new n m = MRbm <$> unsafeIOToPrim (Core.new n m)

zipWithM ::
     PrimMonad m
  => (ℂ -> ℂ -> m ℂ)
  -> MRbm (PrimState m)
  -> MRbm (PrimState m)
  -> MRbm (PrimState m)
  -> m ()
zipWithM func x y z = do
  withVisible x $ \ax -> withVisible y $ \ay -> withVisible z $ \az ->
    zipWithVectorM func ax ay az
  withHidden x $ \bx -> withHidden y $ \by -> withHidden z $ \bz ->
    zipWithVectorM func bx by bz
  withWeights x $ \wx -> withWeights y $ \wy -> withWeights z $ \wz ->
    zipWithMatrixM func wx wy wz

mapM ::
     PrimMonad m
  => (ℂ -> m ℂ)
  -> MRbm (PrimState m)
  -> MRbm (PrimState m)
  -> m ()
mapM func x y = do
  withVisible x $ \ax -> withVisible y $ \ay -> mapVectorM func ax ay
  withHidden x $ \bx -> withHidden y $ \by -> mapVectorM func bx by
  withWeights x $ \wx -> withWeights y $ \wy -> mapMatrixM func wx wy

