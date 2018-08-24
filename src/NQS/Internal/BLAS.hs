{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeApplications #-}

module NQS.Internal.BLAS
  ( AXPY(..)
  , COPY(..)
  , DOTC(..)
  , DOTU(..)
  , NRM2(..)
  , SCAL(..)
  , GEMV(..)
  , HERK(..)
  , fill
  , testGemv
  ) where

import Control.Monad.Primitive
import Control.Exception (assert)
import Data.Complex (Complex)
import Data.Proxy
import Foreign.Storable
import Foreign.Ptr
-- import Foreign.Marshal.Alloc

import Data.Semigroup((<>))
import Numerical.HBLAS.BLAS.FFI ( CBLAS_ORDERT(..)
                                , CBLAS_UPLOT(..)
                                , CBLAS_TRANSPOSET(..)
                                )
import Numerical.HBLAS.BLAS.FFI.Level1
import Numerical.HBLAS.BLAS.FFI.Level2
import Numerical.HBLAS.BLAS.FFI.Level3
import Numerical.HBLAS.Constants
import Numerical.HBLAS.UtilsFFI

import Data.Vector.Storable (Vector, MVector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV

import Data.Singletons

import Lens.Micro((^.))

import NQS.Internal.Types
import NQS.Internal.FFI


instance ToNative Transpose CBLAS_TRANSPOSET where
  encode NoTranspose     = CBLAS_TransposeT 111
  encode Transpose       = CBLAS_TransposeT 112
  encode ConjTranspose   = CBLAS_TransposeT 113
  encode ConjNoTranspose = CBLAS_TransposeT 114

instance ToNative Orientation CBLAS_ORDERT where
  encode Row    = CBOInt 101
  encode Column = CBOInt 102

instance ToNative MatUpLo CBLAS_UPLOT where
  {-# INLINE encode #-}
  encode MatUpper = CBlasUPLO 121
  encode MatLower = CBlasUPLO 122

type ScalFun scale el m = scale -> MDenseVector 'Direct (PrimState m) el -> m ()

type AxpyFun el m = el -> MDenseVector 'Direct (PrimState m) el
                       -> MDenseVector 'Direct (PrimState m) el -> m ()

type DotFun el m res = MDenseVector 'Direct (PrimState m) el
                    -> MDenseVector 'Direct (PrimState m) el
                    -> m res

type CopyFun el m = MDenseVector 'Direct (PrimState m) el
                 -> MDenseVector 'Direct (PrimState m) el
                 -> m ()

type Nrm2Fun el m res = MDenseVector 'Direct (PrimState m) el -> m res

type GemvFun orient el m = Transpose
                        -> el -> MDenseMatrix orient (PrimState m) el
                        -> MDenseVector 'Direct (PrimState m) el
                        -> el -> MDenseVector 'Direct (PrimState m) el
                        -> m ()

type HerkFun scale el orient m
  = MatUpLo
 -> Transpose
 -> scale -> MDenseMatrix orient (PrimState m) el
 -> scale -> MDenseMatrix orient (PrimState m) el
 -> m ()

-- | Fills a vector with a given value.
fill ::
     (PrimMonad m, Storable el, COPY (MDenseVector 'Direct) el)
  => MDenseVector 'Direct (PrimState m) el -> el -> m ()
fill v x =
  V.unsafeThaw (V.singleton x) >>= \t -> copy (MDenseVector (v ^. dim)  0 t) v


{-
        SCAL
  ================
-}

class SCAL scale v el where
  scal :: PrimMonad m => scale -> v (PrimState m) el -> m ()

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
    doScal alpha (MDenseVector dim stride buffer)
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

instance SCAL Float (MDenseVector 'Direct) Float where
  scal = scalAbstraction "sscal" cblas_sscal_safe cblas_sscal_unsafe (\x f -> f x)

instance SCAL Float (MDenseVector 'Direct) (Complex Float) where
  scal = scalAbstraction "csscal" cblas_csscal_safe cblas_csscal_unsafe (\x f -> f x)

instance SCAL (Complex Float) (MDenseVector 'Direct) (Complex Float) where
  scal = scalAbstraction "cscal" cblas_cscal_safe cblas_cscal_unsafe withRStorable_

instance SCAL Double (MDenseVector 'Direct) Double where
  scal = scalAbstraction "dscal" cblas_dscal_safe cblas_dscal_unsafe (\x f -> f x)

instance SCAL Double (MDenseVector 'Direct) (Complex Double) where
  scal = scalAbstraction "zdscal" cblas_zdscal_safe cblas_zdscal_unsafe (\x f -> f x)

instance SCAL (Complex Double) (MDenseVector 'Direct) (Complex Double) where
  scal = scalAbstraction "zscal" cblas_zscal_safe cblas_zscal_unsafe withRStorable_


{-
        AXPY
  ================
-}


class AXPY v el where
  axpy :: PrimMonad m => el -> v (PrimState m) el -> v (PrimState m) el -> m ()

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
    doAxpy alpha (MDenseVector xDim xStride xBuff)
                 (MDenseVector yDim yStride yBuff)
      | not (isValidVector xDim xStride (MV.length xBuff)) = error $!
        badVectorInfo axpyName "X" xDim xStride (MV.length xBuff)
      | not (isValidVector yDim yStride (MV.length yBuff)) = error $!
        badVectorInfo axpyName "Y" yDim yStride (MV.length yBuff)
      | xDim /= yDim = error $! axpyName <> ": Inconsistent dimensions: " <>
                                show xDim <> " != " <> show yDim <> "."
      | otherwise =
        unsafeWithPrim xBuff $ \xPtr ->
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

instance AXPY (MDenseVector 'Direct) Float where
  axpy = axpyAbstraction "saxpy" cblas_saxpy_safe cblas_saxpy_unsafe (\x f -> f x)

instance AXPY (MDenseVector 'Direct) Double where
  axpy = axpyAbstraction "daxpy" cblas_daxpy_safe cblas_daxpy_unsafe (\x f -> f x)

instance AXPY (MDenseVector 'Direct) (Complex Float) where
  axpy = axpyAbstraction "caxpy" cblas_caxpy_safe cblas_caxpy_unsafe withRStorable_

instance AXPY (MDenseVector 'Direct) (Complex Double) where
  axpy = axpyAbstraction "zaxpy" cblas_zaxpy_safe cblas_zaxpy_unsafe withRStorable_


{-
        CDOT
  ================
-}

class DOTU v el res where
  dotu :: PrimMonad m => v (PrimState m) el -> v (PrimState m) el -> m res

{-# SPECIALISE dotu :: PrimMonad m => DotFun Float m Float #-}
{-# SPECIALISE dotu :: PrimMonad m => DotFun Float m Double #-}
{-# SPECIALISE dotu :: PrimMonad m => DotFun Double m Double #-}
{-# SPECIALISE dotu :: PrimMonad m => DotFun (Complex Float) m (Complex Float) #-}
{-# SPECIALISE dotu :: PrimMonad m => DotFun (Complex Double) m (Complex Double) #-}

class DOTC v el res where
  dotc :: PrimMonad m => v (PrimState m) el -> v (PrimState m) el -> m res

{-# SPECIALISE dotc :: PrimMonad m => DotFun Float m Float #-}
{-# SPECIALISE dotc :: PrimMonad m => DotFun Float m Double #-}
{-# SPECIALISE dotc :: PrimMonad m => DotFun Double m Double #-}
{-# SPECIALISE dotc :: PrimMonad m => DotFun (Complex Float) m (Complex Float) #-}
{-# SPECIALISE dotc :: PrimMonad m => DotFun (Complex Double) m (Complex Double) #-}

realDotAbstraction ::
     (PrimMonad m, Storable el, Num el, Show el)
  => String
  -> NoScalarDotFunFFI el res
  -> NoScalarDotFunFFI el res
  -> DotFun el m res
{-# NOINLINE realDotAbstraction #-}
realDotAbstraction dotName dotSafeFFI dotUnsafeFFI = doDot
  where
    shouldCallFast :: Int -> Bool
    shouldCallFast n = flopsThreshold >= fromIntegral n
    doDot (MDenseVector xDim xStride xBuff)
          (MDenseVector yDim yStride yBuff)
      | not (isValidVector xDim xStride (MV.length xBuff)) = error $!
        badVectorInfo dotName "X" xDim xStride (MV.length xBuff)
      | not (isValidVector yDim yStride (MV.length yBuff)) = error $!
        badVectorInfo dotName "Y" yDim yStride (MV.length yBuff)
      | xDim /= yDim = error $! dotName <> ": Inconsistent dimensions: " <>
                                show xDim <> " != " <> show yDim <> "."
      | otherwise =
        unsafeWithPrim xBuff $ \xPtr ->
          unsafeWithPrim yBuff $ \yPtr ->
            unsafePrimToPrim $!
              (if shouldCallFast xDim
                 then dotUnsafeFFI
                 else dotSafeFFI)
                (fromIntegral xDim)
                xPtr (fromIntegral xStride)
                yPtr (fromIntegral yStride)

complexDotAbstraction ::
     (PrimMonad m, Storable el, Num el, Show el)
  => String
  -> ComplexDotFunFFI el
  -> ComplexDotFunFFI el
  -> DotFun el m el
{-# NOINLINE complexDotAbstraction #-}
complexDotAbstraction dotName dotSafeFFI dotUnsafeFFI = doDot
  where
    shouldCallFast :: Int -> Bool
    shouldCallFast n = flopsThreshold >= fromIntegral n
    doDot (MDenseVector xDim xStride xBuff)
          (MDenseVector yDim yStride yBuff)
      | not (isValidVector xDim xStride (MV.length xBuff)) = error $!
        badVectorInfo dotName "X" xDim xStride (MV.length xBuff)
      | not (isValidVector yDim yStride (MV.length yBuff)) = error $!
        badVectorInfo dotName "Y" yDim yStride (MV.length yBuff)
      | xDim /= yDim = error $! dotName <> ": Inconsistent dimensions: " <>
                                show xDim <> " != " <> show yDim <> "."
      | otherwise =
        unsafeWithPrim xBuff $ \xPtr ->
          unsafeWithPrim yBuff $ \yPtr ->
            unsafePrimToPrim $!
              alloca $ \outPtr -> do
                poke outPtr 0
                (if shouldCallFast xDim
                   then dotUnsafeFFI
                   else dotSafeFFI)
                  (fromIntegral xDim)
                  xPtr (fromIntegral xStride)
                  yPtr (fromIntegral yStride)
                  outPtr
                peek outPtr

instance DOTC (MDenseVector 'Direct) Float Float where
  dotc = realDotAbstraction "sdot" cblas_sdot_safe cblas_sdot_unsafe

instance DOTC (MDenseVector 'Direct) Double Double where
  dotc = realDotAbstraction "ddot" cblas_ddot_safe cblas_ddot_unsafe

instance DOTC (MDenseVector 'Direct) Float Double where
  dotc = realDotAbstraction "dsdot" cblas_dsdot_safe cblas_dsdot_unsafe

instance DOTC (MDenseVector 'Direct) (Complex Float) (Complex Float) where
  dotc = complexDotAbstraction "cdotc" cblas_cdotc_safe cblas_cdotc_unsafe

instance DOTC (MDenseVector 'Direct) (Complex Double) (Complex Double) where
  dotc = complexDotAbstraction "zdotc" cblas_zdotc_safe cblas_zdotc_unsafe

instance DOTU (MDenseVector 'Direct) Float Float where
  dotu = realDotAbstraction "sdot" cblas_sdot_safe cblas_sdot_unsafe

instance DOTU (MDenseVector 'Direct) Double Double where
  dotu = realDotAbstraction "ddot" cblas_ddot_safe cblas_ddot_unsafe

instance DOTU (MDenseVector 'Direct) Float Double where
  dotu = realDotAbstraction "dsdot" cblas_dsdot_safe cblas_dsdot_unsafe

instance DOTU (MDenseVector 'Direct) (Complex Float) (Complex Float) where
  dotu = complexDotAbstraction "cdotu" cblas_cdotu_safe cblas_cdotu_unsafe

instance DOTU (MDenseVector 'Direct) (Complex Double) (Complex Double) where
  dotu = complexDotAbstraction "zdotu" cblas_zdotu_safe cblas_zdotu_unsafe


class COPY v el where
  copy :: PrimMonad m => v (PrimState m) el -> v (PrimState m) el -> m ()

{-# SPECIALISE copy :: PrimMonad m => CopyFun Float m #-}
{-# SPECIALISE copy :: PrimMonad m => CopyFun Double m #-}
{-# SPECIALISE copy :: PrimMonad m => CopyFun (Complex Float) m #-}
{-# SPECIALISE copy :: PrimMonad m => CopyFun (Complex Double) m #-}

copyAbstraction ::
     (PrimMonad m, Storable el, Num el, Show el)
  => String
  -> CopyFunFFI el
  -> CopyFunFFI el
  -> CopyFun el m
{-# NOINLINE copyAbstraction #-}
copyAbstraction copyName copySafeFFI copyUnsafeFFI = doCopy
  where
    shouldCallFast :: Int -> Bool
    shouldCallFast _ = True
    doCopy (MDenseVector xDim xStride xBuff)
           (MDenseVector yDim yStride yBuff)
      | not (isValidVector xDim xStride (MV.length xBuff)) = error $!
        badVectorInfo copyName "X" xDim xStride (MV.length xBuff)
      | not (isValidVector yDim yStride (MV.length yBuff)) = error $!
        badVectorInfo copyName "Y" yDim yStride (MV.length yBuff)
      | xDim /= yDim = error $! copyName <> ": Inconsistent dimensions: " <>
                                show xDim <> " != " <> show yDim <> "."
      | otherwise =
        unsafeWithPrim xBuff $ \xPtr ->
          unsafeWithPrim yBuff $ \yPtr ->
            unsafePrimToPrim $!
              (if shouldCallFast xDim
                 then copyUnsafeFFI
                 else copySafeFFI)
                (fromIntegral xDim)
                xPtr (fromIntegral xStride)
                yPtr (fromIntegral yStride)

instance COPY (MDenseVector 'Direct) Float where
  copy = copyAbstraction "scopy" cblas_scopy_safe cblas_scopy_unsafe

instance COPY (MDenseVector 'Direct) Double where
  copy = copyAbstraction "dcopy" cblas_dcopy_safe cblas_dcopy_unsafe

instance COPY (MDenseVector 'Direct) (Complex Float) where
  copy = copyAbstraction "ccopy" cblas_ccopy_safe cblas_ccopy_unsafe

instance COPY (MDenseVector 'Direct) (Complex Double) where
  copy = copyAbstraction "zcopy" cblas_zcopy_safe cblas_zcopy_unsafe


class NRM2 v el res | v el -> res where
  nrm2 :: PrimMonad m => v (PrimState m) el -> m res

{-# SPECIALISE nrm2 :: PrimMonad m => Nrm2Fun Float m Float #-}
{-# SPECIALISE nrm2 :: PrimMonad m => Nrm2Fun Double m Double #-}
{-# SPECIALISE nrm2 :: PrimMonad m => Nrm2Fun (Complex Float) m Float #-}
{-# SPECIALISE nrm2 :: PrimMonad m => Nrm2Fun (Complex Double) m Double #-}

nrm2Abstraction ::
     (PrimMonad m, Storable el, Num el, Show el)
  => String
  -> Nrm2FunFFI el res
  -> Nrm2FunFFI el res
  -> Nrm2Fun el m res
{-# NOINLINE nrm2Abstraction #-}
nrm2Abstraction nrm2Name nrm2SafeFFI nrm2UnsafeFFI = doNrm2
  where
    shouldCallFast :: Int -> Bool
    shouldCallFast n = fromIntegral n <= flopsThreshold
    doNrm2 (MDenseVector xDim xStride xBuff)
      | not (isValidVector xDim xStride (MV.length xBuff)) = error $!
        badVectorInfo nrm2Name "X" xDim xStride (MV.length xBuff)
      | otherwise =
        unsafeWithPrim xBuff $ \xPtr ->
            unsafePrimToPrim $!
              (if shouldCallFast xDim
                 then nrm2UnsafeFFI
                 else nrm2SafeFFI) (fromIntegral xDim) xPtr (fromIntegral xStride)

instance NRM2 (MDenseVector 'Direct) Float Float where
  nrm2 = nrm2Abstraction "snrm2" cblas_snrm2_safe cblas_snrm2_unsafe

instance NRM2 (MDenseVector 'Direct) Double Double where
  nrm2 = nrm2Abstraction "dnrm2" cblas_dnrm2_safe cblas_dnrm2_unsafe

instance NRM2 (MDenseVector 'Direct) (Complex Float) Float where
  nrm2 = nrm2Abstraction "scnrm2" cblas_scnrm2_safe cblas_scnrm2_unsafe

instance NRM2 (MDenseVector 'Direct) (Complex Double) Double where
  nrm2 = nrm2Abstraction "dznrm2" cblas_dznrm2_safe cblas_dznrm2_unsafe



class GEMV mat v el where
  gemv :: PrimMonad m
       => Transpose
       -> el -> mat (PrimState m) el -> v (PrimState m) el
       -> el -> v (PrimState m) el -> m ()

gemvAbstraction :: forall m el scale orient.
     (PrimMonad m, Storable el, Show el, SingI orient)
  => String
  -> GemvFunFFI scale el
  -> GemvFunFFI scale el
  -> (el -> (scale -> m ()) -> m ())
  -> GemvFun orient el m
{-# NOINLINE gemvAbstraction #-}
gemvAbstraction gemvName gemvSafeFFI gemvUnsafeFFI constHandler = doGemv
  where
    shouldCallFast :: Int -> Int -> Bool
    shouldCallFast !n !m = flopsThreshold >= (fromIntegral n) * (fromIntegral m)
    doGemv trans alpha a@(MDenseMatrix aRows aCols aStride aBuff)
                       x@(MDenseVector xDim xStride xBuff)
                  beta y@(MDenseVector yDim yStride yBuff) =
      assertValid gemvName "A" a $
      assertValid gemvName "X" x $
      assertValid gemvName "Y" y $
        unsafeWithPrim aBuff $ \aPtr ->
        unsafeWithPrim xBuff $ \xPtr ->
        unsafeWithPrim yBuff $ \yPtr ->
        constHandler alpha $ \alphaPtr ->
        constHandler beta $ \betaPtr ->
          do
            unsafePrimToPrim $!
              (if shouldCallFast xDim yDim then gemvUnsafeFFI else gemvSafeFFI)
                (encode (orientationOf a)) (encode trans)
                (fromIntegral aRows) (fromIntegral aCols)
                alphaPtr
                aPtr (fromIntegral aStride)
                xPtr (fromIntegral xStride)
                betaPtr
                yPtr (fromIntegral yStride)

instance SingI orient =>
  GEMV (Mutable (DenseMatrix orient)) (MDenseVector 'Direct) Float where
    gemv = gemvAbstraction "sgemv" cblas_sgemv_safe cblas_sgemv_unsafe (\x f -> f x)

instance SingI orient =>
  GEMV (Mutable (DenseMatrix orient)) (MDenseVector 'Direct) Double where
    gemv = gemvAbstraction "dgemv" cblas_dgemv_safe cblas_dgemv_unsafe (\x f -> f x)

instance SingI orient =>
  GEMV (Mutable (DenseMatrix orient)) (MDenseVector 'Direct) (Complex Float) where
    gemv = gemvAbstraction "cgemv" cblas_cgemv_safe cblas_cgemv_unsafe withRStorable_

instance SingI orient =>
  GEMV (Mutable (DenseMatrix orient)) (MDenseVector 'Direct) (Complex Double) where
    gemv = gemvAbstraction "zgemv" cblas_zgemv_safe cblas_zgemv_unsafe withRStorable_

testGemv :: IO ()
testGemv = do
  aBuff <- newVectorAligned 12 64 :: IO (MV.IOVector Double)
  V.copy aBuff (V.fromList [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  let a = MDenseMatrix @'Row 4 3 3 aBuff
  print (orientationOf a)
  xBuff <- newVectorAligned 4 64 :: IO (MV.IOVector Double)
  V.copy xBuff (V.fromList [1, 2, 3, 4])
  let x = MDenseVector @'Direct 3 1 xBuff
  yBuff <- newVectorAligned 3 64 :: IO (MV.IOVector Double)
  -- V.copy yBuff (V.fromList [1, 2, 3])
  let y = MDenseVector @'Direct 3 1 yBuff
  gemv Transpose 1.0 a x 0.0 y
  print =<< V.unsafeFreeze yBuff


class HERK mat scale el where
  herk :: PrimMonad m
       => MatUpLo -> Transpose
       -> scale -> mat (PrimState m) el
       -> scale -> mat (PrimState m) el
       -> m ()

herkAbstraction
  :: forall m el scale scalePtr. (Storable el, PrimMonad m)
  => String
  -> HerkFunFFI scalePtr el
  -> HerkFunFFI scalePtr el
  -> (scale -> (scalePtr -> m ()) -> m ())
  -> (forall orient . SingI orient => HerkFun scale el orient m)
{-# NOINLINE herkAbstraction #-}
herkAbstraction herkName herkSafeFFI herkUnsafeFFI constHandler = doHerk
 where
  isBadHerk :: Transpose -> Int -> Int -> Int -> Int -> Bool
  isBadHerk NoTranspose   ax ay cx cy = (ax /= cx)
  isBadHerk ConjTranspose ax ay cx cy = (ay /= cx)
  isBadHerk trans _ _ _ _ =
    error $! herkName <> ": trans " <> show trans <> " is not supported."
  doHerk :: forall orient. (SingI orient) => HerkFun scale el orient m
  doHerk uplo trans alpha a@(MDenseMatrix ax ay astride abuff) beta c@(MDenseMatrix cx cy cstride cbuff)
    | cx /= cy = error $!
      herkName <> ": C must be a square matrix, but has dimensions " <> show (cx, cy) <> "."
    | isBadHerk trans ax ay cx cy = error $!
      herkName <> ": Incompatible dimensions: ax ay cx cy trans: " <> show [ax, ay, cx, cy]
               <> " " <> show trans
    | MV.overlaps abuff cbuff = error $! herkName <> ": Input and output buffers overlap."
    | otherwise =
      withMatrixFFI a $ \layout aRows aCols aPtr aStride ->
      withMatrixFFI c $ \_      cRows _     cPtr cStride ->
      constHandler alpha $ \alphaPtr ->
      constHandler beta $ \betaPtr ->
        let k = if (trans == NoTranspose) then aCols else aRows
        in  unsafePrimToPrim $!
          herkSafeFFI ((encode . fromSing) (sing :: Sing orient))
                      (encode uplo)
                      (encode trans)
                      cRows
                      k
                      alphaPtr
                      aPtr
                      aStride
                      betaPtr
                      cPtr
                      cStride

instance SingI orient => HERK (Mutable (DenseMatrix orient)) Float (Complex Float) where
  herk = herkAbstraction "cherk" cblas_cherk_safe cblas_cherk_unsafe (\x f -> f x)

instance SingI orient => HERK (Mutable (DenseMatrix orient)) Double (Complex Double) where
  herk = herkAbstraction "zherk" cblas_zherk_safe cblas_zherk_unsafe (\x f -> f x)
