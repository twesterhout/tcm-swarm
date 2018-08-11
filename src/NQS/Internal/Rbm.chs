{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}


module NQS.Internal.Rbm
  ( Hamiltonian(..)
  , _RbmC'new
  , _RbmC'sample
  , _RbmC'sampleMoments
  , _RbmC'getVisible
  , _RbmC'getHidden
  , _RbmC'getWeights
  , _RbmC'size
  , _RbmC'sizeVisible
  , _RbmC'sizeHidden
  ) where

import Control.DeepSeq
import Control.Exception (assert)
import Control.Monad (when, (>=>))
import Control.Monad.Primitive
import Data.Complex
import Data.Semigroup ((<>))
import Foreign.Storable.Complex
import Data.Coerce
import Data.Proxy
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable.Mutable (MVector(..), IOVector)
import qualified Data.Vector.Storable.Mutable as MV
import Foreign
import Foreign.C.Error
import Foreign.C.Types
import GHC.Generics (Generic)
import System.IO.Unsafe (unsafePerformIO)

import Lens.Micro
import Lens.Micro.Extras

import NQS.Internal.Types

#include "nqs.h"


{#pointer *tcm_Complex8 as Tcm_Complex8Ptr -> Complex CFloat #}
{#pointer *tcm_Complex16 as Tcm_Complex16Ptr -> Complex CDouble #}

-- ::

{#pointer *tcm_cRbm as Tcm_cRbmPtr -> `RbmCore (Complex Float)' #}

-- :: 

{#fun unsafe tcm_cRbm_Create
  { `Int', `Int' } -> `Ptr (RbmCore (Complex Float))' id #}


-- foreign import ccall unsafe "nqs.h tcm_cRbm_Destroy"
foreign import ccall unsafe "nqs.h &tcm_cRbm_Destroy"
  tcm_cRbm_Destroy :: FinalizerPtr (RbmCore (Complex Float))

_RbmC'new :: Int -> Int -> IO (ForeignPtr (RbmCore (Complex Float)))
_RbmC'new n m
  | n < 0 || m < 0 = error $!
    "_RbmC'new: RBM dimensions must be non-negative, but got " <>
    show (n, m) <> "."
  | otherwise = tcm_cRbm_Create n m >>= newForeignPtr tcm_cRbm_Destroy

-- :: 

{#enum tcm_Hamiltonian as Hamiltonian
  { HEISENBERG_1D_OPEN as Heisenberg1DOpen,
    HEISENBERG_1D_PERIODIC as Heisenberg1DPeriodic } deriving (Eq, Generic) #}

-- ::

newtype Tcm_Range = Tcm_Range (Int, Int, Int)

{#pointer *tcm_Range as Tcm_RangePtr -> Tcm_Range #}

instance Storable Tcm_Range where
  sizeOf _ = {#sizeof tcm_Range#}
  alignment _ = {#alignof tcm_Range#}
  peek p = Tcm_Range <$> ((,,)
                     <$> (fromIntegral <$> {#get tcm_Range.start#} p)
                     <*> (fromIntegral <$> {#get tcm_Range.stop#} p)
                     <*> (fromIntegral <$> {#get tcm_Range.step#} p))
  poke p (Tcm_Range (start, stop, step)) =
    {#set tcm_Range.start#} p (fromIntegral start) >>
    {#set tcm_Range.stop#} p (fromIntegral stop) >>
    {#set tcm_Range.step#} p (fromIntegral step)

withTcm_Range :: (Int, Int, Int) -> (Ptr Tcm_Range -> IO b) -> IO b
withTcm_Range range@(low, high, step) fun
  | low > high = error $ "Range " <> show range <>
                         "is invalid: lower bound cannot exceed the upper bound."
  | step <= 0  = error $ "Range " <> show range <>
                         "is invalid: only positives steps are currently supported."
  | otherwise = alloca $ \p -> poke p (Tcm_Range range) >> fun p


{#pointer *tcm_cEstimate as Tcm_cEstimatePtr
  -> `Estimate NormalError (Complex Float)' #}

instance Storable (Estimate NormalError (Complex Float)) where
  sizeOf _ = {#sizeof tcm_cEstimate#}
  alignment _ = {#alignof tcm_cEstimate#}
  peek p =
    let fromP x = castPtr $ p `plusPtr` x :: Ptr (Complex Float)
        p1 = fromP {#offsetof tcm_cEstimate->value#}
        p2 = fromP {#offsetof tcm_cEstimate->error#} 
    in Estimate <$> (peek p1) <*> (NormalError <$> (peek p2))
  poke p (Estimate val (NormalError err)) =
    let fromP x = castPtr $ p `plusPtr` x :: Ptr (Complex Float)
        p1 = fromP {#offsetof tcm_cEstimate->value#}
        p2 = fromP {#offsetof tcm_cEstimate->error#}
    in (poke p1 val) >> (poke p2 err)


data Tcm_cTensor1 = Tcm_cTensor1 {-# UNPACK #-}!(Ptr (Complex Float))
                                 {-# UNPACK #-}!Int

{#pointer *tcm_cTensor1 as Tcm_cTensor1Ptr -> Tcm_cTensor1 #}

instance Storable Tcm_cTensor1 where
  sizeOf _ = {#sizeof tcm_cTensor1#}
  alignment _ = {#alignof tcm_cTensor1#}
  peek p =
    let getExtent x = {#get tcm_cTensor1.extents#} x >>= flip peekElemOff 0
    in Tcm_cTensor1 <$> (coerce <$> {#get tcm_cTensor1.data#} p)
                    <*> (fromIntegral <$> getExtent p)
  poke address (Tcm_cTensor1 p n) =
    let setExtent :: Ptr Tcm_cTensor1 -> CInt -> IO ()
        setExtent x = poke $ coerce
                           $ x `plusPtr` {#offsetof tcm_cTensor1.extents#}
    in {#set tcm_cTensor1.data#} address (coerce p) >>
         setExtent address (fromIntegral n)


data Tcm_cTensor2 = Tcm_cTensor2 {-# UNPACK #-}!(Ptr (Complex Float))
                                 {-# UNPACK #-}!Int
                                 {-# UNPACK #-}!Int

{#pointer *tcm_cTensor2 as Tcm_cTensor2Ptr -> Tcm_cTensor2 #}

instance Storable Tcm_cTensor2 where
  sizeOf _ = {#sizeof tcm_cTensor2#}
  alignment _ = {#alignof tcm_cTensor2#}
  peek p =
    let getExtent !i !x = {#get tcm_cTensor2.extents#} x >>= flip peekElemOff i
    in Tcm_cTensor2 <$> (coerce <$> {#get tcm_cTensor2.data#} p)
                    <*> (fromIntegral <$> getExtent 0 p)
                    <*> (fromIntegral <$> getExtent 1 p)
  poke address (Tcm_cTensor2 p n0 n1) =
    let setExtent :: Int -> Ptr Tcm_cTensor2 -> CInt -> IO ()
        setExtent !i !x = flip pokeElemOff i $
          coerce $ x `plusPtr` {#offsetof tcm_cTensor2.extents#}
    in {#set tcm_cTensor1.data#} address (coerce p) >>
         setExtent 0 address (fromIntegral n0) >>
           setExtent 1 address (fromIntegral n1)

allocaPrim :: (Storable a, PrimMonad m, PrimBase min) => (Ptr a -> min b) -> m b
allocaPrim = \f -> unsafePrimToPrim $ alloca (unsafePrimToIO . f)

withAllocaPrim :: (Storable a, PrimMonad m, PrimBase min) => a -> (Ptr a -> min b) -> m b
withAllocaPrim = \x fun -> unsafePrimToPrim $ alloca $ \xPtr ->
  poke xPtr x >> (unsafePrimToIO . fun) xPtr

withAlloca :: (Coercible a b, Storable b) => a -> (Ptr b -> IO c) -> IO c
withAlloca = \x fun -> alloca (\p -> poke p (coerce x) >> fun p)

maybeToPtr :: Storable a => Maybe a -> (Ptr a -> IO b) -> IO b
maybeToPtr (Just x) = withAlloca x
maybeToPtr Nothing = \f -> f nullPtr

withMagnetisation :: Maybe Int -> (Ptr CInt -> IO b) -> IO b
withMagnetisation x = maybeToPtr (fromIntegral <$> x)

mvectorAsTcm_cTensor1 ::
     PrimMonad m
  => MVector (PrimState m) (Complex Float)
  -> (Tcm_cTensor1 -> m b)
  -> m b
mvectorAsTcm_cTensor1 (MVector n fp) = \fun ->
  withForeignPtrPrim fp $ \xPtr -> fun (Tcm_cTensor1 xPtr n)

asTcm_cTensor1 :: PrimMonad m
               => MDenseVector 'Direct (PrimState m) (Complex Float)
               -> (Tcm_cTensor1 -> m b)
               -> m b
asTcm_cTensor1 x@(asTuple -> (n, i, buff))
  | not (isValidVector n i (MV.length buff)) = error $!
    badVectorInfo "asTcm_cTensor1" "vector" n i (MV.length buff)
  | i /= 1 = error $!
    "asTcm_cTensor1: tcm_cTensor1 represents a contiguous vector, but \
    \passed vector has stride " <> show i <> "."
  | otherwise = \fun -> (withForeignPtrPrim . fst . MV.unsafeToForeignPtr0) buff $ \xPtr ->
    fun (Tcm_cTensor1 xPtr n)

asTcm_cTensor2 :: PrimMonad m
               => MDenseMatrix 'Row (PrimState m) (Complex Float)
               -> (Tcm_cTensor2 -> m b)
               -> m b
asTcm_cTensor2 x@(asTuple -> (_, xdim, ydim, i, buff))
  | i /= ydim = error $!
    "asTcm_cTensor2: tcm_cTensor2 represents a contiguous rank-2 tensor, \
    \but passed matrix has stride != #cols: " <> show i <> show ydim <> "."
  | xdim * ydim > MV.length buff = error $!
    "asTcm_cTensor2: Logical number of elements in the matrix (" <>
    show (xdim * ydim) <> "exceeds the length of the underlying buffer (" <>
    show (MV.length buff) <> ")."
  | otherwise = \fun -> (withForeignPtrPrim . fst . MV.unsafeToForeignPtr0) buff $ \xPtr ->
    fun (Tcm_cTensor2 xPtr xdim ydim)

-- void tcm_cRbm_Sample_Moments(tcm_cRbm const* const rbm,
--     tcm_Hamiltonian const hamiltonian, int const number_runs,
--     tcm_Range const* const steps, int const* const magnetisation,
--     tcm_cTensor1* moments);

{#fun tcm_cRbm_Sample_Moments
  { id `Ptr (RbmCore (Complex Float))'
  , `Hamiltonian'
  , `Int'
  , withTcm_Range* `(Int, Int, Int)'
  , withMagnetisation* `Maybe Int'
  , id `Ptr Tcm_cTensor1' } -> `()' #}

_RbmC'sampleMoments :: 
     forall m. PrimMonad m
  => Ptr (RbmCore (Complex Float))
  -> Hamiltonian
  -> Int
  -> (Int, Int, Int)
  -> Maybe Int
  -> Int
  -> m (V.Vector (Complex Float))
_RbmC'sampleMoments rbm hamiltonian numberRuns steps magnetisation n =
  do
    moments <- MV.unsafeNew n
    mvectorAsTcm_cTensor1 moments $ \mT ->
      unsafeIOToPrim $
        withAlloca mT $ \mPtr ->
          tcm_cRbm_Sample_Moments rbm hamiltonian numberRuns steps magnetisation mPtr
    V.unsafeFreeze moments

{#fun tcm_cRbm_Sample
  { id `Ptr (RbmCore (Complex Float))'
  , `Hamiltonian'
  , `Int'
  , withTcm_Range* `(Int, Int, Int)'
  , withMagnetisation* `Maybe Int'
  , alloca- `Estimate NormalError (Complex Float)' peek*
  , id `Ptr Tcm_cTensor1'
  , id `Ptr Tcm_cTensor2' } -> `()' #}

_RbmC'sample :: 
     forall m. PrimMonad m
  => Ptr (RbmCore (Complex Float))
  -> Hamiltonian
  -> Int
  -> (Int, Int, Int)
  -> Maybe Int
  -> MDenseVector 'Direct (PrimState m) (Complex Float)
  -> MDenseMatrix 'Row (PrimState m) (Complex Float)
  -> m (Estimate NormalError (Complex Float))
_RbmC'sample rbm hamiltonian numberRuns steps magnetisation force derivatives =
  asTcm_cTensor1 force $ \fT ->
    asTcm_cTensor2 derivatives $ \dT ->
      unsafeIOToPrim $
        withAlloca fT $ \fPtr ->
          withAlloca dT $ \dPtr ->
            tcm_cRbm_Sample rbm hamiltonian numberRuns steps magnetisation fPtr dPtr

{#fun tcm_cCovariance { id `Ptr Tcm_cTensor2' , id `Ptr Tcm_cTensor2' } -> `()' #}

_RbmC'covariance ::
     forall m. PrimMonad m
  => MDenseMatrix 'Row (PrimState m) (Complex Float)
  -> MDenseMatrix 'Row (PrimState m) (Complex Float)
  -> m ()
_RbmC'covariance derivatives out =
  asTcm_cTensor2 derivatives $ \dT ->
    asTcm_cTensor2 out $ \oT ->
      unsafeIOToPrim $
        withAlloca dT $ \derPtr ->
          withAlloca oT $ \outPtr ->
            tcm_cCovariance derPtr outPtr

{#fun pure unsafe tcm_cRbm_Size as _RbmC'size { `Tcm_cRbmPtr' } -> `Int' #}
{#fun pure unsafe tcm_cRbm_Size_visible as _RbmC'sizeVisible { `Tcm_cRbmPtr' } -> `Int' #}
{#fun pure unsafe tcm_cRbm_Size_hidden as _RbmC'sizeHidden { `Tcm_cRbmPtr' } -> `Int' #}

{-
{#fun tcm_cRbm_Axpy
  { withAllocaValue* `Complex Float', id `Ptr Tcm_cTensor1', `Tcm_cRbmPtr' } -> `()' #}

_RbmC'axpy ::
     forall m. PrimMonad m
  => Complex Float
  -> MDenseVector 'Direct (PrimState m) (Complex Float)
  -> Ptr (RbmCore (Complex Float))
  -> m ()
_RbmC'axpy a x yptr
  | _RbmC'size yptr /= (x ^. dim) = error $
    "_RbmC'axpy: Dimensions do not match: " <> show (x ^. dim) <> " != " <>
    show (_RbmC'size yptr) ++ "."
  | otherwise = unsafeIOToPrim $ asTcm_cTensor1 x $ \xptr -> tcm_cRbm_Axpy a xptr yptr
-}


{#fun tcm_cRbm_Get_visible
  { id `Ptr (RbmCore (Complex Float))' , alloca- `Tcm_cTensor1' peek* } -> `()' #}

{#fun tcm_cRbm_Get_hidden
  { id `Ptr (RbmCore (Complex Float))' , alloca- `Tcm_cTensor1' peek* } -> `()' #}

{#fun tcm_cRbm_Get_weights
  { id `Ptr (RbmCore (Complex Float))' , alloca- `Tcm_cTensor2' peek* } -> `()' #}

_RbmC'getVisible ::
     PrimMonad m
  => Ptr (RbmCore (Complex Float))
  -> m (MVector (PrimState m) (Complex Float))
_RbmC'getVisible rbm = unsafeIOToPrim $ do
  (Tcm_cTensor1 p n) <- tcm_cRbm_Get_visible rbm
  fp <- newForeignPtr_ p
  return $! MV.unsafeFromForeignPtr0 fp n

_RbmC'getHidden ::
     PrimMonad m
  => Ptr (RbmCore (Complex Float))
  -> m (MVector (PrimState m) (Complex Float))
_RbmC'getHidden rbm = unsafeIOToPrim $ do
  (Tcm_cTensor1 p n) <- tcm_cRbm_Get_hidden rbm
  fp <- newForeignPtr_ p
  return $! MV.unsafeFromForeignPtr0 fp n

_RbmC'getWeights ::
     PrimMonad m
  => Ptr (RbmCore (Complex Float))
  -> m (MVector (PrimState m) (Complex Float))
_RbmC'getWeights rbm = unsafeIOToPrim $ do
  (Tcm_cTensor2 p n1 n2) <- tcm_cRbm_Get_weights rbm
  fp <- newForeignPtr_ p
  return $! MV.unsafeFromForeignPtr0 fp (n1 * n2)




