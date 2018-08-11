{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_HADDOCK show-extensions #-}

module NQS.Internal.Types
  ( RbmCore
  , Rbm(..)
  , MRbm(..)
  , Estimate(..)
  , NormalError(..)
  , EnergyMeasurement(..)
  -- , SRMeasurement(..)
  , withRbm
  , withRbmPure
  , withMRbm
  , withMRbmPure
  , withForeignPtrPrim

  , Mutable(..)
  , Variant(..)
  , Orientation(..)
  , DenseVector
  , DenseMatrix
  , MDenseVector
  , MDenseMatrix
  , isValidVector
  , badVectorInfo
  , HasBuffer(..)
  , HasStride(..)
  , HasDim(..)

  , asTuple
  , fromTuple
  , IsBLASVector(..)
  , FreezeThaw(..)

  , HasMean(..)
  , HasVar(..)
  -- , HasEnergy(..)
  -- , HasForce(..)
  -- , HasDerivatives(..)
  ) where

import Foreign.Storable
import Foreign.ForeignPtr

import Control.DeepSeq
import Control.Monad.Primitive
import Data.Complex
import GHC.Generics (Generic)

import Data.Semigroup ((<>))
import Lens.Micro
import Lens.Micro.TH

import System.IO.Unsafe
import Data.Coerce

import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.ForeignPtr.Unsafe

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import Data.Vector.Storable (Vector, MVector)

import qualified Numerical.HBLAS.MatrixTypes as HBLAS
import Numerical.HBLAS.MatrixTypes (Variant(..), Orientation(..))

deriving instance Generic Variant
deriving instance NFData Variant

deriving instance Generic Orientation
deriving instance NFData Orientation

newtype DenseVector variant a = DenseVector (HBLAS.DenseVector variant a)

newtype DenseMatrix orientation a = DenseMatrix (HBLAS.DenseMatrix orientation a)

data family Mutable (v :: * -> *) :: * -> * -> *

data instance Mutable (DenseVector variant) s a = MDenseVector {-# UNPACK #-}!(HBLAS.MDenseVector s variant a)
type MDenseVector variant = Mutable (DenseVector variant)

data instance Mutable (DenseMatrix orientation) s a = MDenseMatrix {-# UNPACK #-}!(HBLAS.MDenseMatrix s orientation a)
type MDenseMatrix orientation = Mutable (DenseMatrix orientation)

-- | Returns whether the range of accesses is valid.
isValidVector :: Int -- ^ Logical dimension
              -> Int -- ^ Stride
              -> Int -- ^ Buffer size
              -> Bool
isValidVector n i size = i > 0 && (n == 0 || n > 0 && (n - 1) * i < size)

badVectorInfo :: String -> String -> Int -> Int -> Int -> String
badVectorInfo funcName argName n i size
  | n < 0 = funcName <> ": " <> argName <>
            " has negative logical dimension: " <> show n <> "."
  | i <= 0 = funcName <> ": " <> argName <>
             " has a non-positive stride: " <> show i <> "."
  | otherwise = funcName <> ": " <> argName <>
                " has invalid range of accesses: dim = " <> show n <>
                ", stride = " <> show i <> ", bufferSize = " <> show size <> "."

class HasBuffer s a | s -> a where
  buffer :: Lens' s a

class HasStride s a | s -> a where
  stride :: Lens' s a

class HasDim s a | s -> a where
  dim :: Lens' s a

class IsTuple s a | s -> a where
  asTuple :: s -> a
  fromTuple :: a -> s


instance Storable a => HasBuffer (DenseVector variant a) (Vector a) where
  buffer inj (DenseVector (HBLAS.DenseVector var dim stride buf)) =
    DenseVector . HBLAS.DenseVector var dim stride <$> inj buf

instance Storable a => HasBuffer (MDenseVector variant s a) (MVector s a) where
  buffer inj (MDenseVector (HBLAS.MutableDenseVector var dim stride buf)) =
    MDenseVector . HBLAS.MutableDenseVector var dim stride <$> inj buf

instance Storable a => HasBuffer (DenseMatrix orientation a) (Vector a) where
  buffer inj (DenseMatrix (HBLAS.DenseMatrix orientation xdim ydim stride buf)) =
    DenseMatrix . HBLAS.DenseMatrix orientation xdim ydim stride <$> inj buf

instance Storable a => HasBuffer (MDenseMatrix orientation s a) (MVector s a) where
  buffer inj (MDenseMatrix (HBLAS.MutableDenseMatrix orientation xdim ydim stride buf)) =
    MDenseMatrix . HBLAS.MutableDenseMatrix orientation xdim ydim stride <$> inj buf


instance Storable a => HasStride (DenseVector variant a) Int where
  stride inj (DenseVector (HBLAS.DenseVector var dim stride buf)) =
    DenseVector . (\x -> HBLAS.DenseVector var dim x buf) <$> inj stride

instance Storable a => HasStride (MDenseVector variant s a) Int where
  stride inj (MDenseVector (HBLAS.MutableDenseVector var dim stride buf)) =
    MDenseVector . (\x -> HBLAS.MutableDenseVector var dim x buf) <$> inj stride

instance Storable a => HasStride (DenseMatrix orientation a) Int where
  stride inj (DenseMatrix (HBLAS.DenseMatrix orientation xdim ydim stride buf)) =
    DenseMatrix . (\x -> HBLAS.DenseMatrix orientation xdim ydim x buf) <$> inj stride

instance Storable a => HasStride (MDenseMatrix orientation s a) Int where
  stride inj (MDenseMatrix (HBLAS.MutableDenseMatrix orientation xdim ydim stride buf)) =
    MDenseMatrix . (\x -> HBLAS.MutableDenseMatrix orientation xdim ydim x buf) <$> inj stride


instance Storable a => HasDim (DenseVector variant a) Int where
  dim inj (DenseVector (HBLAS.DenseVector var dim stride buf)) =
    DenseVector . (\x -> HBLAS.DenseVector var x stride buf) <$> inj dim

instance Storable a => HasDim (MDenseVector variant s a) Int where
  dim inj (MDenseVector (HBLAS.MutableDenseVector var dim stride buf)) =
    MDenseVector . (\x -> HBLAS.MutableDenseVector var x stride buf) <$> inj dim

instance Storable a => HasDim (DenseMatrix orientation a) (Int, Int) where
  dim inj (DenseMatrix (HBLAS.DenseMatrix orientation xdim ydim stride buf)) =
    DenseMatrix . (\(x, y) -> HBLAS.DenseMatrix orientation x y stride buf) <$> inj (xdim, ydim)

instance Storable a => HasDim (MDenseMatrix orientation s a) (Int, Int) where
  dim inj (MDenseMatrix (HBLAS.MutableDenseMatrix orientation xdim ydim stride buf)) =
    MDenseMatrix . (\(x, y) -> HBLAS.MutableDenseMatrix orientation x y stride buf) <$> inj (xdim, ydim)


class IsBLASVector s v | s -> v where
  toBLASVector :: s -> v

instance Storable a => IsTuple (DenseVector 'Direct a) (Int, Int, Vector a) where
  asTuple x = (x ^. dim, x ^. stride, x ^. buffer)
  fromTuple (n, i, b) = DenseVector $! HBLAS.DenseVector HBLAS.SDirect n i b

instance Storable a => IsTuple (MDenseVector 'Direct s a) (Int, Int, MVector s a) where
  asTuple x = (x ^. dim, x ^. stride, x ^. buffer)
  fromTuple (n, i, b) = MDenseVector $! HBLAS.MutableDenseVector HBLAS.SDirect n i b

instance Storable a => IsTuple (DenseMatrix 'Row a) (Orientation, Int, Int, Int, Vector a) where
  asTuple x = let (xdim, ydim) = x ^. dim
               in (Row, xdim, ydim, x ^. stride, x ^. buffer)
  fromTuple (Row, xdim, ydim, stride, b) = DenseMatrix $!
    HBLAS.DenseMatrix HBLAS.SRow xdim ydim stride b

instance Storable a => IsTuple (DenseMatrix 'Column a) (Orientation, Int, Int, Int, Vector a) where
  asTuple x = let (xdim, ydim) = x ^. dim
               in (Column, xdim, ydim, x ^. stride, x ^. buffer)
  fromTuple (Column, xdim, ydim, stride, b) = DenseMatrix $!
    HBLAS.DenseMatrix HBLAS.SColumn xdim ydim stride b

instance Storable a => IsTuple (MDenseMatrix 'Row s a) (Orientation, Int, Int, Int, MVector s a) where
  asTuple x = let (xdim, ydim) = x ^. dim
               in (Row, xdim, ydim, x ^. stride, x ^. buffer)
  fromTuple (Row, xdim, ydim, stride, b) = MDenseMatrix $!
    HBLAS.MutableDenseMatrix HBLAS.SRow xdim ydim stride b

instance Storable a => IsTuple (MDenseMatrix 'Column s a) (Orientation, Int, Int, Int, MVector s a) where
  asTuple x = let (xdim, ydim) = x ^. dim
               in (Column, xdim, ydim, x ^. stride, x ^. buffer)
  fromTuple (Column, xdim, ydim, stride, b) = MDenseMatrix $!
    HBLAS.MutableDenseMatrix HBLAS.SColumn xdim ydim stride b

instance Storable a => IsBLASVector (Int, Int, Vector a) (DenseVector 'Direct a) where
  toBLASVector (dim, stride, xs)
    | dim < 0 || stride <= 0 = error $
      "toBLASVector: BLAS vectors have non-negative length and positive \
      \stride, but got (dim, stride) = " <> show (dim, stride) <> "."
    | dim * stride > V.length xs = error $
      "toBLASVector: dim * stride > length xs: " <> show (dim * stride) <>
      " > " <> show (V.length xs) <> "."
    | otherwise = DenseVector $! HBLAS.DenseVector HBLAS.SDirect dim stride xs

instance Storable a => IsBLASVector (Int, Int, MVector s a) (MDenseVector 'Direct s a) where
  toBLASVector (dim, stride, xs)
    | dim < 0 || stride <= 0 = error $
      "toBLASVector: BLAS vectors have non-negative length and positive \
      \stride, but got (dim, stride) = " <> show (dim, stride) <> "."
    | dim * stride > MV.length xs = error $
      "toBLASVector: dim * stride > length xs: " <> show (dim * stride) <>
      " > " <> show (MV.length xs) <> "."
    | otherwise = MDenseVector $! HBLAS.MutableDenseVector HBLAS.SDirect dim stride xs

class FreezeThaw (v :: * -> *) a where
  unsafeFreeze :: PrimMonad m => (Mutable v) (PrimState m) a -> m (v a)
  unsafeThaw :: PrimMonad m => v a -> m ((Mutable v) (PrimState m) a)

instance Storable a => FreezeThaw (DenseVector variant) a where
  unsafeFreeze (MDenseVector (HBLAS.MutableDenseVector var dim stride mv)) =
    DenseVector . HBLAS.DenseVector var dim stride <$> V.unsafeFreeze mv
  unsafeThaw (DenseVector (HBLAS.DenseVector var dim stride mv)) =
    MDenseVector . HBLAS.MutableDenseVector var dim stride <$> V.unsafeThaw mv

instance Storable a => FreezeThaw (DenseMatrix orientation) a where
  unsafeFreeze (MDenseMatrix (HBLAS.MutableDenseMatrix orientation xdim ydim stride mv)) =
    DenseMatrix . HBLAS.DenseMatrix orientation xdim ydim stride <$> V.unsafeFreeze mv
  unsafeThaw (DenseMatrix (HBLAS.DenseMatrix orientation xdim ydim stride v)) =
    MDenseMatrix . HBLAS.MutableDenseMatrix orientation xdim ydim stride <$> V.unsafeThaw v

data family RbmCore a :: *

data instance RbmCore (Complex Float) = RbmC

data Rbm a = Rbm {-# UNPACK #-}!(ForeignPtr (RbmCore a))

data instance Mutable Rbm s a = MRbm {-# UNPACK #-}!(ForeignPtr (RbmCore a))
type MRbm = (Mutable Rbm)

touchForeignPtrPrim :: PrimMonad m => ForeignPtr a -> m ()
{-# NOINLINE touchForeignPtrPrim #-}
touchForeignPtrPrim fp = unsafeIOToPrim $! touchForeignPtr fp

withForeignPtrPrim :: PrimMonad m => ForeignPtr a -> (Ptr a -> m b) -> m b
{-# INLINE withForeignPtrPrim #-}
withForeignPtrPrim p func = do r <- func (unsafeForeignPtrToPtr p)
                               touchForeignPtrPrim p
                               return r

withRbm :: PrimMonad m => Rbm a -> (Ptr (RbmCore a) -> m b) -> m b
withRbm (Rbm p) func = withForeignPtrPrim p func

withRbmPure :: Rbm a -> (Ptr (RbmCore a) -> b) -> b
withRbmPure (Rbm p) func = unsafePerformIO $! withForeignPtrPrim p (return . func)

withMRbm :: PrimMonad m => MRbm (PrimState m) a -> (Ptr (RbmCore a) -> m b) -> m b
withMRbm (MRbm p) func = withForeignPtrPrim p func

withMRbmPure :: MRbm s a -> (Ptr (RbmCore a) -> b) -> b
withMRbmPure (MRbm p) func = unsafePerformIO $! withForeignPtr p (return . func)



instance FreezeThaw Rbm a where
  unsafeFreeze (MRbm fp) = return $! Rbm fp
  unsafeThaw (Rbm fp) = return $! MRbm fp


data Estimate e a = Estimate
  { estPoint :: !a
  , estError :: !(e a)
  } deriving (Generic, NFData)

newtype NormalError a = NormalError a
  deriving (Eq, Ord, Read, Show, Generic, NFData)

data EnergyMeasurement a = EnergyMeasurement
  { _energyMeasurementMean :: !a
  , _energyMeasurementVar :: !a
  } deriving (Generic)

makeFields ''EnergyMeasurement

