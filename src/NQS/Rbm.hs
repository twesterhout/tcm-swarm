{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module NQS.Rbm
  ( Rbm(..)
  , mkRbm
  , size
  , sizeVisible
  , sizeHidden
  , sizeWeights
  , map
  , mapM
  , zipWith
  , zipWithM
  ) where

import Prelude hiding (length, zipWith, zipWithM, map, mapM)

import Data.Semigroup((<>))
import Data.Aeson
import Control.Monad((>=>))
import Control.Monad.Identity (Identity(..))
import Control.Monad.Primitive
import Control.Monad.ST

import Data.Vector.Storable (Vector, length)
import qualified Data.Vector.Storable as V

import NQS.Internal.Rbm (Rbm(..))
import qualified NQS.Internal.Rbm as Core
import qualified NQS.Rbm.Mutable as Mutable
import NQS.Internal.BLAS
import NQS.Internal.Types


size :: Rbm -> Int
size (Rbm x) = Core.size x

sizeVisible :: Rbm -> Int
sizeVisible (Rbm x) = Core.sizeVisible x

sizeHidden :: Rbm -> Int
sizeHidden (Rbm x) = Core.sizeHidden x

sizeWeights :: Rbm -> Int
sizeWeights (Rbm x) = Core.sizeWeights x

withVisible :: Rbm -> (DenseVector 'Direct ℂ -> r) -> r
withVisible (Rbm x) f = runST $ Core.withVisible x (unsafeFreeze >=> (return . f))

withHidden :: Rbm -> (DenseVector 'Direct ℂ -> r) -> r
withHidden (Rbm x) f = runST $ Core.withHidden x (unsafeFreeze >=> (return . f))

withWeights :: Rbm -> (DenseMatrix 'Column ℂ -> r) -> r
withWeights (Rbm x) f = runST $ Core.withWeights x (unsafeFreeze >=> (return . f))

mkRbm :: Vector ℂ -> Vector ℂ -> Vector ℂ -> Rbm
mkRbm a b w
  | length a * length b /= length w = error $!
    "createRbm: Incompatible dimensions: " <> show (length a, length b, length w)
  | otherwise = let n = length a
                    m = length b
                    lift v = MDenseVector (length v) 1 <$> V.unsafeThaw v
                 in runST $ do rbm <- Mutable.new n m
                               Mutable.setVisible rbm =<< MDenseVector n 1 <$> V.unsafeThaw a
                               Mutable.setHidden rbm =<< MDenseVector m 1 <$> V.unsafeThaw b
                               Mutable.setWeights rbm =<< (MDenseMatrix m n n <$> V.unsafeThaw w
                                 :: ST s (MDenseMatrix 'Row s ℂ))
                               Core.unsafeFreezeRbm rbm

zipWithM :: Monad m => (ℂ -> ℂ -> m ℂ) -> Rbm -> Rbm -> m Rbm
zipWithM func a b = undefined

zipWith :: (ℂ -> ℂ -> ℂ) -> Rbm -> Rbm -> Rbm
zipWith func a b = runIdentity $ zipWithM (\a b -> return $ func a b) a b

mapM :: Monad m => (ℂ -> m ℂ) -> Rbm -> m Rbm
mapM func x = undefined

map :: (ℂ -> ℂ) -> Rbm -> Rbm
map func = runIdentity . mapM (return . func)

instance Num Rbm where
  (+) x y = zipWith (+) x y
  (-) x y = zipWith (-) x y
  (*) x y = zipWith (*) x y
  abs x = map abs x
  signum x = map signum x
  negate x = map negate x
  fromInteger = undefined


instance ToJSON Rbm where
  toJSON rbm =
    withVisible rbm $ \a ->
    withHidden rbm $ \b ->
    withWeights rbm $ \w ->
      object ["visible" .= a, "hidden" .= b, "weights"  .= w]
  toEncoding rbm =
    withVisible rbm $ \a ->
    withHidden rbm $ \b ->
    withWeights rbm $ \w ->
      pairs ("visible" .= a <> "hidden" .= b <> "weights"  .= w)

instance FromJSON Rbm where
  parseJSON = withObject "Rbm" $ \v ->
    mkRbm <$> v .: "visible"
          <*> v .: "hidden"
          <*> (V.concat <$> v .: "weights")
{-
instance IsRbm a => Scalable a (Rbm a) where
  scale λ = map (*λ)

instance (PrimMonad m, DeltaWell m Float Float)
  => DeltaWell m Float (Rbm Float) where
    upDeltaWell κ p x = zipWithM (upDeltaWell κ) p x
-}

