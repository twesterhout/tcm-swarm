{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}

module NQS.Internal.LAPACK
  ( chesv
  , zhesv
  , cgelsd
  , test
  ) where

import           Data.Coerce
import           Data.Complex                   ( Complex )
import           Data.Semigroup                 ( (<>) )
import           Data.Singletons
import           Data.Vector.Storable.Mutable   ( MVector(..) )
import qualified Data.Vector.Storable.Mutable  as MV
import           Data.Vector.Storable           ( Vector(..) )
import qualified Data.Vector.Storable          as V
import           Control.Monad.Primitive
import           Foreign.C.String               ( castCharToCChar )
import           Foreign.C.Types
import           Foreign.Ptr                    ( Ptr
                                                , castPtr
                                                )
import           Foreign.Storable               ( Storable(..) )

import           NQS.Internal.Types
import           NQS.Internal.FFI

#include <mkl_types.h>
#include <mkl_lapacke.h>


instance ToNative MatUpLo CChar where
  {-# INLINE encode #-}
  encode MatUpper = castCharToCChar 'U'
  encode MatLower = castCharToCChar 'L'

hesvCheckArgs ::
     (Storable a, Integral b)
  => MDenseMatrix orient s a
  -> MDenseMatrix orient s a
  -> MVector s b
  -> c
  -> c
hesvCheckArgs a@(MDenseMatrix aRows aCols aStride _)
              b@(MDenseMatrix bRows bCols bStride _)
              ipiv@(MVector ipivSize _)
  | aRows /= aCols = error $! "?hesv: Matrix A is not diagonal: " <> show (aRows, aCols) <> "."
  | aRows /= bRows = error $! "?hesv: A and B have incompatible dimensions: " <>
                             show (aRows, aCols) <> " and " <> show (bRows, bCols) <> "."
  | ipivSize /= aRows = error $! "?hesv: ipiv has wrong size: " <> show ipivSize <>
                                ", but should be " <> show aRows <> "."
  | otherwise = id

hesvCheckErrno :: Monad m => CInt -> m ()
hesvCheckErrno errno
  | errno < 0 = error $! "?hesv: Parameter #" <> show (-errno) <> " is invalid."
  | errno > 0 = error $! "?hesv: The factorization has been completed, \
                        \but D_{" <> (show errno) <> "," <> (show errno) <>
                        "} is 0, so the solution could not be computed."
  | otherwise = return ()

type HesvFun orient m a = MatUpLo
                       -> MDenseMatrix orient (PrimState m) a
                       -> MDenseMatrix orient (PrimState m) a
                       -> MVector (PrimState m) CInt
                       -> m ()

hesvAbstraction ::
     forall orient m a a'. (SingI orient, PrimMonad m, Storable a, RealFrac a, Coercible a a')
  => HesvFunFFI (Complex a')
  -> HesvFunFFI (Complex a')
  -> HesvFun orient m (Complex a)
hesvAbstraction hesv_safe hesv_unsafe uplo a b ipiv@(MVector _ ipivBuff) =
  hesvCheckArgs a b ipiv $
    withMatrixFFI a $ \layout n _    aPtr ldimA ->
    withMatrixFFI b $ \_      _ nrhs bPtr ldimB ->
    withForeignPtr ipivBuff $ \ipivPtr ->
      unsafeIOToPrim $
        hesv_safe layout (encode uplo) n nrhs aPtr ldimA ipivPtr bPtr ldimB >>=
          hesvCheckErrno

chesv :: (PrimMonad m, SingI orient) => HesvFun orient m (Complex Float)
chesv = hesvAbstraction c_chesv_safe c_chesv_unsafe

zhesv :: (PrimMonad m, SingI orient) => HesvFun orient m (Complex Double)
zhesv = hesvAbstraction c_zhesv_safe c_zhesv_unsafe


test :: IO ()
test = do
  a <- V.thaw $ V.fromList [1, 2, 3,
                            0, 1, 4,
                            0, 0, 1] :: IO (MVector (PrimState IO) (Complex Float))
  b <- V.thaw $ V.fromList [4,
                            4,
                            4] :: IO (MVector (PrimState IO) (Complex Float))
  ipiv <- V.thaw $ V.fromList [0, 0, 0] :: IO (MVector (PrimState IO) CInt)
  let a' = MDenseMatrix 3 3 3 a :: MDenseMatrix 'Column (PrimState IO) (Complex Float)
      b' = MDenseMatrix 3 1 3 b :: MDenseMatrix 'Column (PrimState IO) (Complex Float)
  print =<< chesv MatLower a' b' ipiv
  print =<< V.freeze a
  print =<< V.freeze b
  return ()


type HesvFunFFI a = CInt          -- ^ Matrix layout
                 -> CChar         -- ^ Upper/Lower
                 -> CInt          -- ^ n (A is an nxn matrix)
                 -> CInt          -- ^ number of columns in B
                 -> Ptr a -> CInt -- ^ A and leading dimension of A
                 -> Ptr CInt      -- ^ ipiv
                 -> Ptr a -> CInt -- ^ B and leading dimension of B
                 -> IO CInt       -- ^ Status code

foreign import ccall safe "LAPACKE_chesv"
  c_chesv_safe :: HesvFunFFI (Complex CFloat)

foreign import ccall safe "LAPACKE_zhesv"
  c_zhesv_safe :: HesvFunFFI (Complex CDouble)

foreign import ccall unsafe "LAPACKE_chesv"
  c_chesv_unsafe :: HesvFunFFI (Complex CFloat)

foreign import ccall unsafe "LAPACKE_zhesv"
  c_zhesv_unsafe :: HesvFunFFI (Complex CDouble)

-- |
-- @
--     lapack_int LAPACKE_zgelsd(int matrix_layout, lapack_int m, lapack_int n,
--         lapack_int nrhs, lapack_complex_double* a, lapack_int lda,
--         lapack_complex_double* b, lapack_int ldb, double* s, double rcond,
--         lapack_int* rank);
-- @
type GelsdFunFFI real complex
  = CInt                -- ^ Matrix layout
 -> CInt                -- ^ m (A is an mxn matrix)
 -> CInt                -- ^ n (A is an mxn matrix)
 -> CInt                -- ^ nrhs (B is an mxnrhs matrix)
 -> Ptr complex -> CInt -- ^ A and leading dimension of A
 -> Ptr complex -> CInt -- ^ B and leading dimension of B
 -> Ptr real            -- ^ s
 -> real                -- ^ rcond
 -> Ptr CInt            -- ^ Rank of A
 -> IO CInt             -- ^ Status code

foreign import ccall safe "LAPACKE_sgelsd"
  c_sgelsd_safe :: GelsdFunFFI CFloat CFloat

foreign import ccall unsafe "LAPACKE_sgelsd"
  c_sgelsd_unsafe :: GelsdFunFFI CFloat CFloat

foreign import ccall safe "LAPACKE_dgelsd"
  c_dgelsd_safe :: GelsdFunFFI CDouble CDouble

foreign import ccall unsafe "LAPACKE_dgelsd"
  c_dgelsd_unsafe :: GelsdFunFFI CDouble CDouble

foreign import ccall safe "LAPACKE_cgelsd"
  c_cgelsd_safe :: GelsdFunFFI CFloat (Complex CFloat)

foreign import ccall unsafe "LAPACKE_cgelsd"
  c_cgelsd_unsafe :: GelsdFunFFI CFloat (Complex CFloat)

foreign import ccall safe "LAPACKE_zgelsd"
  c_zgelsd_safe :: GelsdFunFFI CDouble (Complex CDouble)

foreign import ccall unsafe "LAPACKE_zgelsd"
  c_zgelsd_unsafe :: GelsdFunFFI CDouble (Complex CDouble)

type GelsdFun orient m real complex
  = MDenseMatrix orient (PrimState m) complex -- ^ A
 -> MDenseMatrix orient (PrimState m) complex -- ^ B
 -> real                                      -- ^ rcond
 -> m Int                                     -- ^ Rank of A

gelsdAbstraction
  :: forall orient m real complex real' complex'
   . ( SingI orient
     , PrimMonad m
     , Storable real, Storable real'
     , Storable complex
     , Coercible real real'
     , Coercible complex complex'
     )
  => GelsdFunFFI real' complex'
  -> GelsdFunFFI real' complex'
  -> GelsdFun orient m real complex
gelsdAbstraction gelsd_safe gelsd_unsafe a b rcond =
  withMatrixFFI a $ \layout n m    aPtr aStride ->
  withMatrixFFI b $ \_      _ nrhs bPtr bStride ->
  allocaArray (fromIntegral (max 1 (min n m))) $ \sPtr ->
  alloca $ \rankPtr ->
    unsafeIOToPrim $ do
      gelsd_safe layout m n nrhs aPtr aStride bPtr bStride sPtr (coerce rcond) rankPtr >>=
        hesvCheckErrno
      fromIntegral <$> peek rankPtr

cgelsd :: (PrimMonad m, SingI orient) => GelsdFun orient m Float (Complex Float)
cgelsd = gelsdAbstraction c_cgelsd_safe c_cgelsd_unsafe

