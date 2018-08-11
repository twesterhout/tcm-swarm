{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}


module PSO.Internal.Neural
  ( Measurement(..)
  , RbmC(..)
  , _RbmC'construct
  , _RbmC'clone
  , _RbmC'getWeights
  , _RbmC'getVisible
  , _RbmC'getHidden
  , _RbmC'heisenberg1D
  -- , _RbmC'plus
  -- , _RbmC'minus
  -- , _RbmC'multiply
  -- , _RbmC'divide
  -- , _RbmC'negate
  ) where

import Control.DeepSeq
import Control.Exception (assert)
import Control.Monad (when, (>=>))
import Control.Monad.Primitive
import Data.Complex
import Data.Coerce
import qualified Data.Vector.Storable as V
import Foreign
import Foreign.C.Error
import Foreign.C.Types
import GHC.Generics (Generic)
import System.IO.Unsafe (unsafePerformIO)

#include "neural.h"

type MKLSize = {#type tcm_size_type #}
type MKLInt  = {#type tcm_difference_type #}

{#pointer *tcm_Complex8 as ComplexCFloatPtr -> Complex CFloat #}

{#pointer *tcm_Complex16 as ComplexCDoublePtr -> Complex CDouble #}


data RbmC = RbmC
  { _rbmC'weightsR    :: {-# UNPACK #-}!(Ptr CFloat)
  , _rbmC'weightsI    :: {-# UNPACK #-}!(Ptr CFloat)
  , _rbmC'visibleR    :: {-# UNPACK #-}!(Ptr CFloat)
  , _rbmC'visibleI    :: {-# UNPACK #-}!(Ptr CFloat)
  , _rbmC'hiddenR     :: {-# UNPACK #-}!(Ptr CFloat)
  , _rbmC'hiddenI     :: {-# UNPACK #-}!(Ptr CFloat)
  , _rbmC'sizeVisible :: {-# UNPACK #-}!Int
  , _rbmC'sizeHidden  :: {-# UNPACK #-}!Int
  } deriving (Generic)

_rbmC'sizeWeights :: RbmC -> Int
_rbmC'sizeWeights rbm = _rbmC'sizeVisible rbm * _rbmC'sizeHidden rbm

{#pointer *tcm_cRBM as RbmCPtr -> RbmC #}

instance Storable RbmC where
  sizeOf _ = {#sizeof tcm_cRBM#}
  alignment _ = {#alignof tcm_cRBM#}
  peek p = RbmC <$> {#get struct tcm_cRBM->_real_weights #} p
                <*> {#get struct tcm_cRBM->_imag_weights #} p
                <*> {#get struct tcm_cRBM->_real_visible #} p
                <*> {#get struct tcm_cRBM->_imag_visible #} p
                <*> {#get struct tcm_cRBM->_real_hidden  #} p
                <*> {#get struct tcm_cRBM->_imag_hidden  #} p
                <*> (fromIntegral <$> {#get struct tcm_cRBM->_size_visible #} p)
                <*> (fromIntegral <$> {#get struct tcm_cRBM->_size_hidden  #} p)
  poke p x = do
    {#set struct tcm_cRBM->_real_weights #} p (_rbmC'weightsR x)
    {#set struct tcm_cRBM->_imag_weights #} p (_rbmC'weightsI x)
    {#set struct tcm_cRBM->_real_visible #} p (_rbmC'visibleR x)
    {#set struct tcm_cRBM->_imag_visible #} p (_rbmC'visibleI x)
    {#set struct tcm_cRBM->_real_hidden  #} p (_rbmC'hiddenR  x)
    {#set struct tcm_cRBM->_imag_hidden  #} p (_rbmC'hiddenI  x)
    {#set struct tcm_cRBM->_size_visible #} p (fromIntegral . _rbmC'sizeVisible $ x)
    {#set struct tcm_cRBM->_size_hidden  #} p (fromIntegral . _rbmC'sizeHidden $ x)


checkCError :: CInt -> IO ()
checkCError err = when (err /= (0 :: CInt)) $ throwErrno "PSO.Internal.Neural"

unsafeWith :: Storable a => V.Vector a -> (Ptr a -> Int -> IO b) -> IO b
unsafeWith v f = let n = V.length v in V.unsafeWith v (\p -> f p n)

-- _RbmC'getWeights ::
--      PrimMonad m
--   => ForeignPtr RbmC -> m (V.MVector (PrimState m) Float, V.MVector (PrimState m) Float)
_RbmC'getWeights = _RbmC'getImpl _rbmC'sizeWeights _rbmC'weightsR _rbmC'weightsI

_RbmC'getVisible = _RbmC'getImpl _rbmC'sizeVisible _rbmC'visibleR _rbmC'visibleI

_RbmC'getHidden = _RbmC'getImpl _rbmC'sizeHidden _rbmC'hiddenR _rbmC'hiddenI

_RbmC'getImpl ::
     PrimMonad m
  => (RbmC -> Int)
  -> (RbmC -> Ptr CFloat)
  -> (RbmC -> Ptr CFloat)
  -> ForeignPtr RbmC -> m (V.MVector (PrimState m) Float, V.MVector (PrimState m) Float)
_RbmC'getImpl size real imag = unsafeIOToPrim .
  flip withForeignPtr (peek >=> \rbm -> coerce <$> pair (size rbm) rbm)
  where
    pair !n rbm = (,) <$> (V.MVector n <$> (newForeignPtr_ . real $ rbm))
                      <*> (V.MVector n <$> (newForeignPtr_ . imag $ rbm))

_RbmC'preAlloc :: (Ptr RbmC -> IO ()) -> IO (ForeignPtr RbmC)
_RbmC'preAlloc f = do
  rbm <- mallocForeignPtr :: IO (ForeignPtr RbmC)
  withForeignPtr rbm f
  addForeignPtrFinalizer _RbmC'destruct rbm
  return rbm

{#fun unsafe tcm_cRBM_Construct as _RbmC'constructImpl
  { `Int', `Int', `RbmCPtr' } -> `()' #}

foreign import ccall unsafe "neural.h &tcm_cRBM_Destruct"
  _RbmC'destruct :: FunPtr (Ptr RbmC -> IO ())

_RbmC'construct :: Int -> Int -> IO (ForeignPtr RbmC)
_RbmC'construct n m = assert (n > 0 && m > 0) $
  _RbmC'preAlloc $ _RbmC'constructImpl n m

{#fun unsafe tcm_cRBM_Clone as _RbmC'cloneImpl
  { withForeignPtr* `ForeignPtr RbmC', `RbmCPtr' } -> `()' #}

_RbmC'clone :: ForeignPtr RbmC -> IO (ForeignPtr RbmC)
_RbmC'clone x = _RbmC'preAlloc $ _RbmC'cloneImpl x


{-
{#fun unsafe tcm_cRBM_Set_weights as _RbmC'setWeightsImpl
  { `RbmCPtr', `ComplexCFloatPtr', `Int' } -> `()' #}

_RbmC'setWeights :: Ptr RbmC -> V.Vector (Complex Float) -> IO ()
_RbmC'setWeights rbm weights = do
  n <- _rbmC'sizeVisible <$> peek rbm
  m <- _rbmC'sizeHidden <$> peek rbm
  if n * m == V.length weights
    then unsafeWith (coerce weights) (_RbmC'setWeightsImpl rbm)
    else error "_RbmC'setWeights: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Set_visible as _RbmC'setVisibleImpl
  { `RbmCPtr', `ComplexCFloatPtr', `Int' } -> `()' #}

_RbmC'setVisible :: Ptr RbmC -> V.Vector (Complex Float) -> IO ()
_RbmC'setVisible rbm visible = do
  n <- _rbmC'sizeVisible <$> peek rbm
  if n == V.length visible
    then unsafeWith (coerce visible) (_RbmC'setVisibleImpl rbm)
    else error "_RbmC'setVisible: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Set_hidden as _RbmC'setHiddenImpl
  { `RbmCPtr', `ComplexCFloatPtr', `Int' } -> `()' #}

_RbmC'setHidden :: Ptr RbmC -> V.Vector (Complex Float) -> IO ()
_RbmC'setHidden rbm hidden = do
  m <- _rbmC'sizeHidden <$> peek rbm
  if m == V.length hidden
    then unsafeWith (coerce hidden) (_RbmC'setHiddenImpl rbm)
    else error "_RbmC'setHidden: Incompatible dimensions."

_RbmC'eqDim :: Ptr RbmC -> Ptr RbmC -> Bool
_RbmC'eqDim x y = unsafePerformIO $
  (&&) <$> ((==) <$> (_rbmC'sizeVisible <$> peek x) <*> (_rbmC'sizeVisible <$> peek y))
       <*> ((==) <$> (_rbmC'sizeHidden <$> peek x) <*> (_rbmC'sizeHidden <$> peek y))

_RbmC'binOp :: (Ptr RbmC -> Ptr RbmC -> Ptr RbmC -> IO ())
            -> Ptr RbmC -> Ptr RbmC -> IO (ForeignPtr RbmC)
_RbmC'binOp cFunc x y
  | x `_RbmC'eqDim` y = _RbmC'preAlloc $ cFunc x y
  | otherwise = error "_RbmC'binOp: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Plus as _RbmC'plusImpl
  { withForeignPtr* `ForeignPtr RbmC', `RbmCPtr', `RbmCPtr' } -> `()' #}

_RbmC'plus :: Ptr RbmC -> Ptr RbmC -> IO (ForeignPtr RbmC)
_RbmC'plus = _RbmC'binOp _RbmC'plusImpl

{#fun unsafe tcm_cRBM_Minus as _RbmC'minusImpl
  { `RbmCPtr', `RbmCPtr', `RbmCPtr' } -> `()' #}

_RbmC'minus :: Ptr RbmC -> Ptr RbmC -> IO (ForeignPtr RbmC)
_RbmC'minus = _RbmC'binOp _RbmC'minusImpl

{#fun unsafe tcm_cRBM_Multiply as _RbmC'multiplyImpl
  { `RbmCPtr', `RbmCPtr', `RbmCPtr' } -> `()' #}

_RbmC'multiply :: Ptr RbmC -> Ptr RbmC -> IO (ForeignPtr RbmC)
_RbmC'multiply = _RbmC'binOp _RbmC'multiplyImpl

{#fun unsafe tcm_cRBM_Divide as _RbmC'divideImpl
  { `RbmCPtr', `RbmCPtr', `RbmCPtr' } -> `()' #}

_RbmC'divide :: Ptr RbmC -> Ptr RbmC -> IO (ForeignPtr RbmC)
_RbmC'divide = _RbmC'binOp _RbmC'divideImpl

{#fun unsafe tcm_cRBM_Negate as _RbmC'negateImpl
  { `RbmCPtr', `RbmCPtr' } -> `()' #}

_RbmC'negate :: Ptr RbmC -> IO (ForeignPtr RbmC)
_RbmC'negate x = _RbmC'preAlloc $ _RbmC'negateImpl x
-}


data Measurement a = Measurement { _measurementMean :: !a, _measurementVar :: !a }
  deriving (Show, Generic)

instance NFData a => NFData (Measurement a)

{#pointer *tcm_sMeasurement as MeasurementS -> `Measurement Float' #}

instance Storable (Measurement Float) where
  sizeOf _ = {#sizeof tcm_sMeasurement#}
  alignment _ = {#alignof tcm_sMeasurement#}
  peek p = coerce <$> Measurement <$> {#get tcm_sMeasurement->mean#} p
                                  <*> {#get tcm_sMeasurement->var#} p
  poke p (Measurement m v) = {#set tcm_sMeasurement->mean#} p (coerce m) >>
                               {#set tcm_sMeasurement->var#} p (coerce v)


{#fun unsafe tcm_cRBM_heisenberg_1d as _RbmC'heisenberg1D
  { withForeignPtr* `ForeignPtr RbmC', `Int', `Int', alloca- `Measurement Float' peek* } -> `()' #}



