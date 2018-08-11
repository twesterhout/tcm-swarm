{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}

module NQS.Internal.LAPACK
  ( test
  , chesv
  ) where

import Control.Monad.Primitive
import Data.Complex
import Data.Proxy
import Foreign.Storable
import Foreign.Storable.Complex
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.C.Types
import Foreign.Marshal.Alloc
import GHC.Generics (Generic)
import Control.DeepSeq
import Data.Coerce
import Data.Vector.Storable.Mutable (MVector(..))
import qualified Data.Vector.Storable.Mutable as MV
import Data.Vector.Storable (Vector(..))
import qualified Data.Vector.Storable as V

import Data.Semigroup ((<>))
import Numerical.HBLAS.Constants

import Data.Char

import NQS.Internal.Types

#include <mkl_types.h>
#include <mkl_lapacke.h>


data MatUpLo = MatUpper | MatLower
  deriving (Eq, Generic, NFData)

encodeUpLo :: MatUpLo -> CChar
encodeUpLo MatUpper = fromIntegral 'U'
encodeUpLo MatLower = fromIntegral 'L'

{-
typedef enum {
    MKL_ROW_MAJOR = 101,
    MKL_COL_MAJOR = 102
} MKL_LAYOUT;
-}
encodeOrientation :: Orientation -> CInt
encodeOrientation Row = 101
encodeOrientation Column = 102

mklAlignment :: Int
mklAlignment = 64



allocaWorkspace :: forall a m min. (Storable a, PrimMonad m, PrimBase min)
                => Int -> (Ptr a -> min b) -> m b
allocaWorkspace n = doAllocaWorkspace undefined
  where doAllocaWorkspace :: a' -> (Ptr a' -> min b) -> m b
        doAllocaWorkspace dummy =
          let size = n * (sizeOf dummy)
              alignment = max mklAlignment (alignmentOf dummy)
           in \f -> unsafeIOToPrim $ allocaBytesAligned size alignment (unsafePrimToIO . f)



type HesvFunFFI a = CInt -> CChar -> CInt -> CInt
                 -> Ptr a -> CInt
                 -> Ptr CInt
                 -> Ptr a -> CInt
                 -> Ptr a -> CInt
                 -> IO CInt

hesvCheckArgs ::
     (PrimMonad m, Storable a, Integral b)
  => Proxy m
  -> MDenseMatrix (PrimState m) orient a
  -> MDenseMatrix (PrimState m) orient a
  -> MVector (PrimState m) b
  -> IO ()
hesvCheckArgs _ (MutableDenseMatrix layout aXDim aYDim aStride _)
                (MutableDenseMatrix _      bXDim bYDim bStride _)
                (MVector ipivSize _)
  | aXDim /= aYDim = error $ "?hesv: Matrix A is not diagonal: shape A = " <>
                             show (aXDim, aYDim) <> "."
  | aXDim /= bXDim = error $ "?hesv: A and B have incompatible dimensions: " <>
                             show (aXDim, aYDim) <> " and " <> show (bXDim, bYDim) <> "."
  | ipivSize /= aXDim = error $ "?hesv: ipiv has wrong size: " <> show ipivSize <>
                                ", but should be " <> show aXDim <> "."
  | otherwise = return ()

hesvCheckErrno :: Monad m => CInt -> m ()
hesvCheckErrno errno
  | errno < 0 = error $ "?hesv: Parameter #" <> show (-errno) <> " is invalid."
  | errno > 0 = error $ "?hesv: The factorization has been completed, \
                        \but D_{" <> (show errno) <> "," <> (show errno) <>
                        "} is 0, so the solution could not be computed."
  | otherwise = return ()

type HesvFun orient m a = MatUpLo
                       -> MDenseMatrix (PrimState m) orient a
                       -> MDenseMatrix (PrimState m) orient a
                       -> MVector (PrimState m) CInt
                       -> m ()

hesvAbstraction ::
     forall m a (orient :: Orientation). (PrimMonad m, Storable a, RealFrac a)
  => HesvFunFFI (Complex a)
  -> HesvFunFFI (Complex a)
  -> HesvFun orient m (Complex a)
hesvAbstraction hesv_safe
                hesv_unsafe
                uplo
                a@(MutableDenseMatrix layout _ n aStride (MVector _ aFgnPtr))
                b@(MutableDenseMatrix _ _ nrhs bStride (MVector _ bFgnPtr))
                ipiv@(MVector _ ipivFgnPtr) =
  let !cLAYOUT = encodeOrientation layout
      !cUPLO = encodeUpLo uplo
      !cN = fromIntegral n
      !cNRHS = fromIntegral nrhs
      !cLDA = fromIntegral aStride
      !cLDB = fromIntegral bStride
      queryWorkSize :: IO Int
      queryWorkSize = alloca $ \workPtr -> hesv_unsafe cLAYOUT cUPLO cN cNRHS
        nullPtr cLDA nullPtr nullPtr cLDB workPtr (-1) >>=
          hesvCheckErrno >> truncate . realPart <$> peek workPtr
  in unsafeIOToPrim $ hesvCheckArgs (undefined :: Proxy m) a b ipiv >>
    queryWorkSize >>= \workSize ->
      allocaBytesAligned (sizeOf (undefined :: Complex a) * workSize)
                         mklAlignment $ \workPtr ->
        withForeignPtr (coerce aFgnPtr) $ \aPtr ->
          withForeignPtr (coerce bFgnPtr) $ \bPtr ->
            withForeignPtr (coerce ipivFgnPtr) $ \ipivPtr ->
              hesv_safe cLAYOUT cUPLO cN cNRHS aPtr cLDA ipivPtr
                bPtr cLDB workPtr (fromIntegral workSize) >>= hesvCheckErrno

chesv :: PrimMonad m => HesvFun orient m (Complex Float)
chesv = hesvAbstraction c_chesv_work_safe c_chesv_work_unsafe

zhesv :: PrimMonad m => HesvFun orient m (Complex Double)
zhesv = hesvAbstraction c_zhesv_work_safe c_zhesv_work_unsafe


test :: IO ()
test = do
  let a1 = V.fromList [1, 2, 3, 0, 1, 4, 0, 0, 1] :: Vector (Complex Float)
      b1 = V.fromList [4, 4, 4] :: Vector (Complex Float)
      ipiv1 = V.fromList [0, 0, 0] :: Vector CInt
  a1m <- V.thaw a1
  b1m <- V.thaw b1
  ipiv1m <- V.thaw ipiv1
  let a = MutableDenseMatrix SRow 3 3 3 a1m
      b = MutableDenseMatrix SRow 3 1 1 b1m
  _ <- chesv MatUpper a b ipiv1m
  print =<< V.freeze a1m
  print =<< V.freeze b1m
  return ()


{-
LAPACK_DECL
lapack_int LAPACKE_chesv_work( int matrix_layout, char uplo, lapack_int n,
                               lapack_int nrhs, lapack_complex_float* a,
                               lapack_int lda, lapack_int* ipiv,
                               lapack_complex_float* b, lapack_int ldb,
                               lapack_complex_float* work, lapack_int lwork );
-}

foreign import ccall safe "LAPACKE_chesv_work"
  c_chesv_work_safe :: HesvFunFFI (Complex Float)

foreign import ccall safe "LAPACKE_zhesv_work"
  c_zhesv_work_safe :: HesvFunFFI (Complex Double)

foreign import ccall unsafe "LAPACKE_chesv_work"
  c_chesv_work_unsafe :: HesvFunFFI (Complex Float)

foreign import ccall unsafe "LAPACKE_zhesv_work"
  c_zhesv_work_unsafe :: HesvFunFFI (Complex Double)


