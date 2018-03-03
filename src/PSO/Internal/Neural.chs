{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}


module PSO.Internal.Neural
  ( Measurement(..)
  , RbmC(..)
  , RbmZ(..)
  , _newRbmC
  , _newRbmZ
  , _cloneRbmC
  , _cloneRbmZ
  , _sizeVisibleC
  , _sizeVisibleZ
  , _sizeHiddenC
  , _sizeHiddenZ
  , _setWeightsC
  , _setWeightsZ
  , _setVisibleC
  , _setVisibleZ
  , _setHiddenC
  , _setHiddenZ
  , _printRbmC
  , _printRbmZ
  , _caxpbyRbmC
  , _zaxpbyRbmZ
  , _cscaleRbmC
  , _zscaleRbmZ
  , McmcC
  , McmcZ
  , _newMcmcC
  , _newMcmcZ
  , _logWFC
  , _logWFZ
  , _logQuotientWF1C
  , _logQuotientWF1Z
  , _logQuotientWF2C
  , _logQuotientWF2Z
  , _propose1C
  , _propose1Z
  , _propose2C
  , _propose2Z
  , _accept1C
  , _accept1Z
  , _accept2C
  , _accept2Z
  , _printMcmcC
  , _printMcmcZ
  , _locEHH1DOpenC
  , _locEHH1DOpenZ
  , _upDeltaWellC
  , _upDeltaWellZ
  , _mcmcBlockC
  ) where


import Foreign.C
import Foreign.Storable
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Marshal
import Data.Complex
import Data.Coerce
import GHC.Generics (Generic)
import Control.DeepSeq
import qualified Data.Vector.Storable as V
import System.IO.Unsafe (unsafePerformIO)

#include "neural.h"

-- type MKLSize = {#type tcm_size_type #}
-- type MKLInt  = {#type tcm_difference_type #}

{#pointer *tcm_cRBM  as RbmC  foreign finalizer tcm_cRBM_Destroy  newtype#}
-- dummy :: Int -- removing this line confuses Vim's syntax highlighting.

{#pointer *tcm_zRBM  as RbmZ  foreign finalizer tcm_zRBM_Destroy  newtype#}
-- dummy :: Int -- removing this line confuses Vim's syntax highlighting.

{#pointer *tcm_cMCMC as McmcC foreign finalizer tcm_cMCMC_Destroy newtype#}
-- dummy :: Int -- removing this line confuses Vim's syntax highlighting.

{#pointer *tcm_zMCMC as McmcZ foreign finalizer tcm_zMCMC_Destroy newtype#}
-- dummy :: Int -- removing this line confuses Vim's syntax highlighting.

{#pointer *tcm_Complex8 as ComplexCFloatPtr -> Complex CFloat #}

{#pointer *tcm_Complex16 as ComplexCDoublePtr -> Complex CDouble #}


data Measurement a = Measurement !a !a
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


{#fun unsafe tcm_cRBM_Create as _newRbmC { `Int', `Int' } -> `RbmC' #}

{#fun unsafe tcm_zRBM_Create as _newRbmZ { `Int', `Int' } -> `RbmZ' #}

{#fun unsafe tcm_cMCMC_Create as
  _newMcmcImplC { `RbmC', `ComplexCFloatPtr' } -> `McmcC' #}

{#fun unsafe tcm_zMCMC_Create as
  _newMcmcImplZ { `RbmZ', `ComplexCDoublePtr' } -> `McmcZ' #}

_newMcmcC :: RbmC -> V.Vector (Complex CFloat) -> IO McmcC
_newMcmcC rbm spin = V.unsafeWith spin (_newMcmcImplC rbm)

_newMcmcZ :: RbmZ -> V.Vector (Complex CDouble) -> IO McmcZ
_newMcmcZ rbm spin = V.unsafeWith spin (_newMcmcImplZ rbm)

{#fun pure unsafe tcm_cRBM_Size_visible as _sizeVisibleC { `RbmC' } -> `Int' #}

{#fun pure unsafe tcm_zRBM_Size_visible as _sizeVisibleZ { `RbmZ' } -> `Int' #}

{#fun pure unsafe tcm_cRBM_Size_hidden  as _sizeHiddenC  { `RbmC' } -> `Int' #}

{#fun pure unsafe tcm_zRBM_Size_hidden  as _sizeHiddenZ  { `RbmZ' } -> `Int' #}

{#fun unsafe tcm_cRBM_Print as _printRbmC { `RbmC' } -> `()' #}

{#fun unsafe tcm_zRBM_Print as _printRbmZ { `RbmZ' } -> `()' #}

{#fun unsafe tcm_cMCMC_Print as _printMcmcC { `McmcC' } -> `()' #}

{#fun unsafe tcm_zMCMC_Print as _printMcmcZ { `McmcZ' } -> `()' #}

{#fun unsafe tcm_cRBM_Set_weights as
  _unsafeSetWeightsC { `RbmC', `ComplexCFloatPtr' } -> `()' #}

{#fun unsafe tcm_zRBM_Set_weights as
  _unsafeSetWeightsZ { `RbmZ', `ComplexCDoublePtr' } -> `()' #}

_setWeightsC :: RbmC -> V.Vector (Complex CFloat) -> IO ()
_setWeightsC rbm weights
  | _sizeHiddenC rbm * _sizeVisibleC rbm == V.length weights =
      V.unsafeWith weights (_unsafeSetWeightsC rbm)
  | otherwise = error "_setWeightsC: Incompatible dimensions."

_setWeightsZ :: RbmZ -> V.Vector (Complex CDouble) -> IO ()
_setWeightsZ rbm weights
  | _sizeHiddenZ rbm * _sizeVisibleZ rbm == V.length weights =
      V.unsafeWith weights (_unsafeSetWeightsZ rbm)
  | otherwise = error "_setWeightsZ: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Set_visible as
  _unsafeSetVisibleC { `RbmC', `ComplexCFloatPtr' } -> `()' #}

{#fun unsafe tcm_zRBM_Set_visible as
  _unsafeSetVisibleZ { `RbmZ', `ComplexCDoublePtr' } -> `()' #}

_setVisibleC :: RbmC -> V.Vector (Complex CFloat) -> IO ()
_setVisibleC rbm visible
  | _sizeVisibleC rbm == V.length visible =
      V.unsafeWith visible (_unsafeSetVisibleC rbm)
  | otherwise = error "_setVisibleC: Incompatible dimensions."

_setVisibleZ :: RbmZ -> V.Vector (Complex CDouble) -> IO ()
_setVisibleZ rbm visible
  | _sizeVisibleZ rbm == V.length visible =
      V.unsafeWith visible (_unsafeSetVisibleZ rbm)
  | otherwise = error "_setVisibleZ: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Set_hidden as
  _unsafeSetHiddenC { `RbmC', `ComplexCFloatPtr' } -> `()' #}

{#fun unsafe tcm_zRBM_Set_hidden as
  _unsafeSetHiddenZ { `RbmZ', `ComplexCDoublePtr' } -> `()' #}

_setHiddenC :: RbmC -> V.Vector (Complex CFloat) -> IO ()
_setHiddenC rbm hidden
  | _sizeHiddenC rbm == V.length hidden =
       V.unsafeWith hidden (_unsafeSetHiddenC rbm)
  | otherwise = error "_setHiddenC: Incompatible dimensions."

_setHiddenZ :: RbmZ -> V.Vector (Complex CDouble) -> IO ()
_setHiddenZ rbm hidden
  | _sizeHiddenZ rbm == V.length hidden =
       V.unsafeWith hidden (_unsafeSetHiddenZ rbm)
  | otherwise = error "_setHiddenZ: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Clone as _cloneRbmC { `RbmC' } -> `RbmC' #}

{#fun unsafe tcm_zRBM_Clone as _cloneRbmZ { `RbmZ' } -> `RbmZ' #}

{#fun unsafe tcm_cRBM_caxpby as
  _caxpbyRbmImplC
    { `CFloat', `CFloat', `RbmC', `CFloat', `CFloat', `RbmC' } -> `()' #}

{#fun unsafe tcm_zRBM_zaxpby as
  _zaxpbyRbmImplZ
    { `CDouble', `CDouble', `RbmZ', `CDouble', `CDouble', `RbmZ' } -> `()' #}

_caxpbyRbmC :: Complex CFloat -> RbmC -> Complex CFloat -> RbmC -> IO ()
_caxpbyRbmC (ar :+ ai) x (br :+ bi) y
  | _sizeVisibleC x == _sizeVisibleC y && _sizeHiddenC x == _sizeHiddenC y =
      _caxpbyRbmImplC ar ai x br bi y
  | otherwise = error "_caxpbyRbmC: Incompatible dimensions."

_zaxpbyRbmZ :: Complex CDouble -> RbmZ -> Complex CDouble -> RbmZ -> IO ()
_zaxpbyRbmZ (ar :+ ai) x (br :+ bi) y
  | _sizeVisibleZ x == _sizeVisibleZ y && _sizeHiddenZ x == _sizeHiddenZ y =
      _zaxpbyRbmImplZ ar ai x br bi y
  | otherwise = error "_zaxpbyRbmZ: Incompatible dimensions."

{#fun unsafe tcm_cRBM_cscale as
  _cscaleRbmImplC { `CFloat', `CFloat', `RbmC' } -> `()' #}

{#fun unsafe tcm_zRBM_zscale as
  _zscaleRbmImplZ { `CDouble', `CDouble', `RbmZ' } -> `()' #}

_cscaleRbmC :: Complex CFloat -> RbmC -> IO ()
_cscaleRbmC (ar :+ ai) x = _cscaleRbmImplC ar ai x

_zscaleRbmZ :: Complex CDouble -> RbmZ -> IO ()
_zscaleRbmZ (ar :+ ai) x = _zscaleRbmImplZ ar ai x

{#fun pure unsafe tcm_cMCMC_Log_wf as
    _logWFC { `McmcC', alloca- `Complex CFloat' peek* } -> `()' #}

{#fun pure unsafe tcm_zMCMC_Log_wf as
    _logWFZ { `McmcZ', alloca- `Complex CDouble' peek* } -> `()' #}

{#fun pure unsafe tcm_cMCMC_Log_quotient_wf1 as
    _logQuotientWF1C
      { `McmcC', `Int', alloca- `Complex CFloat' peek* } -> `()' #}

{#fun pure unsafe tcm_zMCMC_Log_quotient_wf1 as
    _logQuotientWF1Z
      { `McmcZ', `Int', alloca- `Complex CDouble' peek* } -> `()' #}

{#fun pure unsafe tcm_cMCMC_Log_quotient_wf2 as
    _logQuotientWF2C
      { `McmcC', `Int', `Int', alloca- `Complex CFloat' peek* } -> `()' #}

{#fun pure unsafe tcm_zMCMC_Log_quotient_wf2 as
    _logQuotientWF2Z
      { `McmcZ', `Int', `Int', alloca- `Complex CDouble' peek* } -> `()' #}

{#fun pure unsafe tcm_cMCMC_Propose1 as
    _propose1C { `McmcC', `Int' } -> `Float' #}

{#fun pure unsafe tcm_zMCMC_Propose1 as
    _propose1Z { `McmcZ', `Int' } -> `Double' #}

{#fun pure unsafe tcm_cMCMC_Propose2 as
    _propose2C { `McmcC', `Int', `Int' } -> `Float' #}

{#fun pure unsafe tcm_zMCMC_Propose2 as
    _propose2Z { `McmcZ', `Int', `Int' } -> `Double' #}

{#fun unsafe tcm_cMCMC_Accept1 as _accept1C { `McmcC', `Int' } -> `()' #}

{#fun unsafe tcm_zMCMC_Accept1 as _accept1Z { `McmcZ', `Int' } -> `()' #}

{#fun unsafe tcm_cMCMC_Accept2 as
    _accept2C { `McmcC', `Int', `Int' } -> `()' #}

{#fun unsafe tcm_zMCMC_Accept2 as
    _accept2Z { `McmcZ', `Int', `Int' } -> `()' #}

{#fun pure unsafe tcm_cHH1DOpen_Local_energy as
    _locEHH1DOpenC { `McmcC', alloca- `Complex CFloat' peek* } -> `()' #}

{#fun pure unsafe tcm_zHH1DOpen_Local_energy as
    _locEHH1DOpenZ { `McmcZ', alloca- `Complex CDouble' peek* } -> `()' #}

{#fun unsafe tcm_cDelta_well_update as
    _upDeltaWellImplC { `Float', `RbmC', `RbmC', id `Ptr CFloat' } -> `()' #}

{#fun unsafe tcm_zDelta_well_update as
    _upDeltaWellImplZ { `Double', `RbmZ', `RbmZ', id `Ptr CDouble' } -> `()' #}

_upDeltaWellC :: Float -> RbmC -> RbmC -> V.Vector Float -> IO ()
_upDeltaWellC k p x rs =
  V.unsafeWith rs (\ v -> _upDeltaWellImplC k p x (coerce v))

_upDeltaWellZ :: Double -> RbmZ -> RbmZ -> V.Vector Double -> IO ()
_upDeltaWellZ k p x rs =
  V.unsafeWith rs (\ v -> _upDeltaWellImplZ k p x (coerce v))

{#fun unsafe tcm_cRBM_mcmc_block as
  _mcmcBlockC { `RbmC', `Int', `Int', alloca- `Measurement Float' peek* } -> `()' #}

