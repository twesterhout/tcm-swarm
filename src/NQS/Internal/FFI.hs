{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}


module NQS.Internal.FFI
  ( withForeignPtr
  , VectorFunFFI
  , MatrixFunFFI
  , withVectorFFI
  , withMatrixFFI
  , alloca
  , allocaArray
  ) where

import           Control.Monad.Primitive
import qualified Data.Primitive.Types           as Prim ( Addr(..) )
import qualified Data.Primitive.ByteArray       as Prim ( MutableByteArray(..)
                                                , newAlignedPinnedByteArray
                                                , mutableByteArrayContents
                                                )


import Data.Coerce
import GHC.Exts (unsafeCoerce#)
import Data.Vector.Storable.Mutable (MVector(..))
import Data.Singletons
import Foreign.ForeignPtr.Unsafe
import Foreign.Storable (Storable(..))
import Foreign.Ptr
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr(..), ForeignPtrContents(..))
import qualified GHC.Ptr as GHC (Ptr(..))

import GHC.Stack

import NQS.Internal.Types

mallocPlainForeignPtrAlignedBytes
  :: (HasCallStack, PrimMonad m) => Int -> Int -> m (ForeignPtr a)
mallocPlainForeignPtrAlignedBytes size align
  | size < 0 = error "mallocPlainForeignPtrAlignedBytes: size must be >= 0"
  | otherwise = Prim.newAlignedPinnedByteArray size align >>= \mbarr ->
    let (Prim.MutableByteArray mbarr') = mbarr
        (Prim.Addr             addr' ) = Prim.mutableByteArrayContents mbarr
    in  return $! ForeignPtr addr' (PlainPtr (unsafeCoerce# mbarr'))

allocaBytesAligned :: PrimMonad m => Int -> Int -> (Ptr a -> m b) -> m b
allocaBytesAligned !size !align f = do
  !mbarr <- Prim.newAlignedPinnedByteArray size align
  let (Prim.Addr addr) = Prim.mutableByteArrayContents mbarr
  x <- f (GHC.Ptr addr)
  touch mbarr
  return x

alloca :: forall m a b. (PrimMonad m, Storable a) => (Ptr a -> m b) -> m b
alloca =
  allocaBytesAligned (sizeOf (undefined :: a)) (alignment (undefined :: a))

allocaArray
  :: forall m a b
   . (HasCallStack, PrimMonad m, Storable a)
  => Int
  -> (Ptr a -> m b)
  -> m b
allocaArray !n
  | n < 0 = error "allocaArray: size must be >= 0"
  | otherwise = allocaBytesAligned (n * sizeOf (undefined :: a))
                                   (alignment (undefined :: a))

touchForeignPtr :: PrimMonad m => ForeignPtr a -> m ()
{-# NOINLINE touchForeignPtr #-}
touchForeignPtr (ForeignPtr _ r) = touch r

withForeignPtr :: PrimMonad m => ForeignPtr a -> (Ptr a -> m b) -> m b
{-# INLINE withForeignPtr #-}
withForeignPtr p func = do
  r <- func (unsafeForeignPtrToPtr p)
  touchForeignPtr p
  return r

mallocVectorAligned :: forall a. Storable a => Int -> Int -> IO (ForeignPtr a)
mallocVectorAligned n alignment =
  mallocPlainForeignPtrAlignedBytes (n * sizeOf (undefined :: a)) alignment


type VectorFunFFI m a b
  = CInt  -- ^ Length of the vector
 -> Ptr a -- ^ The vector itself
 -> CInt  -- ^ Stride
 -> m b

type MatrixFunFFI m a b
  = CInt  -- ^ Matrix layout
 -> CInt  -- ^ Number of rows
 -> CInt  -- ^ Number of columns
 -> Ptr a -- ^ The matrix itself
 -> CInt  -- ^ Leading dimension
 -> m b

withVectorFFI
  :: (PrimMonad m, Storable a, Coercible a a')
  => MDenseVector 'Direct (PrimState m) a
  -> VectorFunFFI m a' b
  -> m b
{-# INLINE withVectorFFI #-}
withVectorFFI !x@(MDenseVector !size !stride !(MVector _ fp)) !func =
  assertValid "" "" x $ withForeignPtr fp $ \p ->
    func (fromIntegral size) (castPtr p) (fromIntegral stride)

withMatrixFFI
  :: forall orient m a a' b
   . (PrimMonad m, Storable a, Coercible a a', SingI orient)
  => MDenseMatrix orient (PrimState m) a
  -> MatrixFunFFI m a' b
  -> m b
{-# INLINE withMatrixFFI #-}
withMatrixFFI !x@(MDenseMatrix !rows !cols !ldim !(MVector _ fp)) !func =
  assertValid "" "" x $ withForeignPtr fp $ \p -> func layout
                                                       (fromIntegral rows)
                                                       (fromIntegral cols)
                                                       (castPtr p)
                                                       (fromIntegral ldim)
  where layout = (encode . fromSing) (sing :: Sing orient)

instance ToNative Orientation CInt where
  encode Row    = 101
  encode Column = 102
