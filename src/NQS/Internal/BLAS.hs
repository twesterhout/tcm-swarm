{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module NQS.Internal.BLAS
  ( SCAL(..)
  , AXPY(..)
  ) where

import Control.Monad.Primitive
import Data.Complex (Complex)
import Foreign.Storable
import Foreign.Ptr
import Foreign.Marshal.Alloc

import Data.Semigroup((<>))
import Numerical.HBLAS.BLAS.FFI.Level1
import Numerical.HBLAS.Constants
import Numerical.HBLAS.UtilsFFI

import Data.Vector.Storable (Vector, MVector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV


import Lens.Micro((^.))

import NQS.Internal.Types

{-
        SCAL
  ================
-}



type ScalFun scale el m = scale -> MDenseVector 'Direct (PrimState m) el -> m ()

class SCAL scale el where
  scal :: PrimMonad m => ScalFun scale el m

{-# SPECIALISE scal :: PrimMonad m => ScalFun Float Float m #-}
{-# SPECIALISE scal :: PrimMonad m => ScalFun Float (Complex Float) m #-}
{-# SPECIALISE scal :: PrimMonad m => ScalFun (Complex Float) (Complex Float) m #-}
{-# SPECIALISE scal :: PrimMonad m => ScalFun Double Double m #-}
{-# SPECIALISE scal :: PrimMonad m => ScalFun Double (Complex Double) m #-}
{-# SPECIALISE scal :: PrimMonad m => ScalFun (Complex Double) (Complex Double) m #-}

scalAbstraction ::
     (Storable el, PrimMonad m, Show el)
  => String
  -> ScalFunFFI scale el
  -> ScalFunFFI scale el
  -> (scaleplain -> (scale -> m ()) -> m ())
  -> ScalFun scaleplain el m
{-# NOINLINE scalAbstraction #-}
scalAbstraction scalName scalSafeFFI scalUnsafeFFI constHandler = doScal
  where
    shouldCallFast :: Int -> Bool
    shouldCallFast !n = flopsThreshold >= fromIntegral n
    doScal alpha x@(asTuple -> (dim, stride, buffer))
      | not (isValidVector dim stride (MV.length buffer)) = error $!
          badVectorInfo scalName "X" dim stride (MV.length buffer)
      | otherwise =
        unsafeWithPrim buffer $ \xPtr ->
          constHandler alpha $ \alphaPtr -> do
            unsafePrimToPrim $!
              (if shouldCallFast dim
                 then scalUnsafeFFI
                 else scalSafeFFI)
                (fromIntegral dim)
                alphaPtr
                xPtr (fromIntegral stride)

instance SCAL Float Float where
  scal = scalAbstraction "sscal" cblas_sscal_safe cblas_sscal_unsafe (\x f -> f x)

instance SCAL Float (Complex Float) where
  scal = scalAbstraction "csscal" cblas_csscal_safe cblas_csscal_unsafe (\x f -> f x)

instance SCAL (Complex Float) (Complex Float) where
  scal = scalAbstraction "cscal" cblas_cscal_safe cblas_cscal_unsafe withRStorable_

instance SCAL Double Double where
  scal = scalAbstraction "dscal" cblas_dscal_safe cblas_dscal_unsafe (\x f -> f x)

instance SCAL Double (Complex Double) where
  scal = scalAbstraction "zdscal" cblas_zdscal_safe cblas_zdscal_unsafe (\x f -> f x)

instance SCAL (Complex Double) (Complex Double) where
  scal = scalAbstraction "zscal" cblas_zscal_safe cblas_zscal_unsafe withRStorable_


{-
        SCAL
  ================
-}

type AxpyFun el m = el -> DenseVector 'Direct el -> MDenseVector 'Direct (PrimState m) el -> m ()

class AXPY el where
  axpy :: PrimMonad m => AxpyFun el m

{-# SPECIALISE axpy :: PrimMonad m => AxpyFun Float m #-}
{-# SPECIALISE axpy :: PrimMonad m => AxpyFun (Complex Float) m #-}
{-# SPECIALISE axpy :: PrimMonad m => AxpyFun Double m #-}
{-# SPECIALISE axpy :: PrimMonad m => AxpyFun (Complex Double) m #-}




axpyAbstraction ::
     (PrimMonad m, Storable el, Show el)
  => String
  -> AxpyFunFFI scale el
  -> AxpyFunFFI scale el
  -> (el -> (scale -> m ()) -> m ())
  -> AxpyFun el m
{-# NOINLINE axpyAbstraction #-}
axpyAbstraction axpyName axpySafeFFI axpyUnsafeFFI constHandler = doAxpy
  where
    shouldCallFast :: Int -> Bool
    shouldCallFast !n = flopsThreshold >= 2 * (fromIntegral n) -- n for a*x, and n for +y
    doAxpy alpha x@(asTuple -> (xDim, xStride, xBuff))
                 y@(asTuple -> (yDim, yStride, yBuff))
      | not (isValidVector xDim xStride (V.length xBuff)) = error $!
        badVectorInfo axpyName "X" xDim xStride (V.length xBuff)
      | not (isValidVector yDim yStride (MV.length yBuff)) = error $!
        badVectorInfo axpyName "Y" yDim yStride (MV.length yBuff)
      | xDim /= yDim = error $! axpyName <> ": Inconsistent dimensions: " <>
                                show xDim <> " != " <> show yDim <> "."
      | otherwise =
        unsafeWithPurePrim xBuff $ \xPtr ->
          unsafeWithPrim yBuff $ \yPtr ->
            constHandler alpha $ \alphaPtr -> do
              unsafePrimToPrim $!
                (if shouldCallFast xDim
                   then axpyUnsafeFFI
                   else axpySafeFFI)
                  (fromIntegral xDim)
                  alphaPtr
                  xPtr (fromIntegral xStride)
                  yPtr (fromIntegral yStride)

instance AXPY Float where
  axpy = axpyAbstraction "saxpy" cblas_saxpy_safe cblas_saxpy_unsafe (\x f -> f x)

instance AXPY Double where
  axpy = axpyAbstraction "daxpy" cblas_daxpy_safe cblas_daxpy_unsafe (\x f -> f x)

instance AXPY (Complex Float) where
  axpy = axpyAbstraction "caxpy" cblas_caxpy_safe cblas_caxpy_unsafe withRStorable_

instance AXPY (Complex Double) where
  axpy = axpyAbstraction "zaxpy" cblas_zaxpy_safe cblas_zaxpy_unsafe withRStorable_


