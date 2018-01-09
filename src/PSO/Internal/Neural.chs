{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ScopedTypeVariables #-}


module PSO.Internal.Neural
  ( RbmC
  , _newRbmC
  , _cloneRbmC
  , _sizeVisibleC
  , _sizeHiddenC
  , _setWeightsC
  , _setVisibleC
  , _setHiddenC
  , _printRbmC
  , _caxpbyRbmC
  , _cscaleRbmC
  , test
  , McmcC
  , _newMcmcC
  , _logWFC
  , _logQuotientWF1C
  , _logQuotientWF2C
  , _proposeC1
  , _proposeC2
  , _acceptC1
  , _acceptC2
  , _printMcmcC
  , _locEHH1DOpenC
  , _upDeltaWellC
  ) where


import Foreign.C
import Foreign.Storable
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Marshal
import Data.Complex
import Data.Coerce
import qualified Data.Vector.Storable as V
import System.IO.Unsafe (unsafePerformIO)

#include "neural.h"

-- type MKLSize = {#type tcm_size_type #}
-- type MKLInt  = {#type tcm_difference_type #}

{#pointer *tcm_cRBM  as RbmC  foreign finalizer tcm_cRBM_Destroy  newtype#}
-- dummy :: Int -- removing this line confuses Vim's syntax highlighting.

{#pointer *tcm_cMCMC as McmcC foreign finalizer tcm_cMCMC_Destroy newtype#}
-- dummy :: Int -- removing this line confuses Vim's syntax highlighting.

-- newtype Complex8 = Complex8 (Complex CFloat)
--   deriving (Storable)

{#pointer *tcm_Complex8 as ComplexCFloatPtr -> Complex CFloat #}

_allocaC :: (Ptr (Complex CFloat) -> IO b) -> IO b
_allocaC = alloca

{#fun unsafe tcm_cRBM_Create as
  _newRbmC { `Int', `Int' } -> `RbmC' #}

{#fun unsafe tcm_cMCMC_Create as
  _newMcmcCImpl { `RbmC', `ComplexCFloatPtr' } -> `McmcC' #}

_newMcmcC :: RbmC -> V.Vector (Complex CFloat) -> IO McmcC
_newMcmcC rbm spin = V.unsafeWith spin (_newMcmcCImpl rbm)


{#fun pure unsafe tcm_cRBM_Size_visible as _sizeVisibleC { `RbmC' } -> `Int' #}
{#fun pure unsafe tcm_cRBM_Size_hidden  as _sizeHiddenC  { `RbmC' } -> `Int' #}

{#fun unsafe tcm_cRBM_Print as _printRbmC { `RbmC' } -> `()' #}

{#fun unsafe tcm_cMCMC_Print as _printMcmcC { `McmcC' } -> `()' #}

{#fun unsafe tcm_cRBM_Set_weights as
  _unsafeSetWeightsC { `RbmC', `ComplexCFloatPtr' } -> `()' #}

_setWeightsC :: RbmC -> V.Vector (Complex CFloat) -> IO ()
_setWeightsC rbm weights
  | _sizeHiddenC rbm * _sizeVisibleC rbm == V.length weights =
      V.unsafeWith weights (_unsafeSetWeightsC rbm)
  | otherwise = error "_setWeightsC: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Set_visible as
  _unsafeSetVisibleC { `RbmC', `ComplexCFloatPtr' } -> `()' #}

_setVisibleC:: RbmC -> V.Vector (Complex CFloat) -> IO ()
_setVisibleC rbm visible
  | _sizeVisibleC rbm == V.length visible =
      V.unsafeWith visible (_unsafeSetVisibleC rbm)
  | otherwise = error "_setVisibleC: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Set_hidden as
  _unsafeSetHiddenC { `RbmC', `ComplexCFloatPtr' } -> `()' #}

_setHiddenC:: RbmC -> V.Vector (Complex CFloat) -> IO ()
_setHiddenC rbm hidden
  | _sizeHiddenC rbm == V.length hidden =
       V.unsafeWith hidden (_unsafeSetHiddenC rbm)
  | otherwise = error "_setHiddenC: Incompatible dimensions."

{#fun unsafe tcm_cRBM_Clone as
  _cloneRbmC { `RbmC' } -> `RbmC' #}

{#fun unsafe tcm_cRBM_caxpby as
  _caxpbyRbmCImpl
    { `CFloat', `CFloat', `RbmC', `CFloat', `CFloat', `RbmC' } -> `()' #}

_caxpbyRbmC :: Complex CFloat -> RbmC -> Complex CFloat -> RbmC -> IO ()
_caxpbyRbmC (ar :+ ai) x (br :+ bi) y
  | _sizeVisibleC x == _sizeVisibleC y && _sizeHiddenC x == _sizeHiddenC y =
      _caxpbyRbmCImpl ar ai x br bi y
  | otherwise = error "_caxpbyRbmC: Incompatible dimensions."

{#fun unsafe tcm_cRBM_cscale as
  _cscaleRbmCImpl { `CFloat', `CFloat', `RbmC' } -> `()' #}

_cscaleRbmC :: Complex CFloat -> RbmC -> IO ()
_cscaleRbmC (ar :+ ai) x = _cscaleRbmCImpl ar ai x

{#fun pure unsafe tcm_cMCMC_Log_wf as
    _logWFC { `McmcC', alloca- `Complex CFloat' peek*} -> `()' #}

-- _logWFC :: McmcC -> Complex CFloat
-- _logWFC mcmc = unsafePerformIO . _unsafeToPure $ _logWFCImpl mcmc

{#fun pure unsafe tcm_cMCMC_Log_quotient_wf1 as
    _logQuotientWF1C
      { `McmcC', `Int', alloca- `Complex CFloat' peek*} -> `()' #}

-- _logQuotientWF1C :: McmcC -> Int -> Complex CFloat
-- _logQuotientWF1C mcmc flip1 = unsafePerformIO . _unsafeToPure $
--   _logQuotientWF1CImpl mcmc flip1

{#fun pure unsafe tcm_cMCMC_Log_quotient_wf2 as
    _logQuotientWF2C
      { `McmcC', `Int', `Int', alloca- `Complex CFloat' peek*} -> `()' #}

-- _logQuotientWF2C :: McmcC -> Int -> Int -> Complex CFloat
-- _logQuotientWF2C mcmc flip1 flip2 = unsafePerformIO . _unsafeToPure $
--   _logQuotientWF2CImpl mcmc flip1 flip2

{#fun pure unsafe tcm_cMCMC_Propose1 as
    _proposeC1 { `McmcC', `Int' } -> `Float' #}

{#fun pure unsafe tcm_cMCMC_Propose2 as
    _proposeC2 { `McmcC', `Int', `Int' } -> `Float' #}

{#fun unsafe tcm_cMCMC_Accept1 as
    _acceptC1 { `McmcC', `Int' } -> `()' #}

{#fun unsafe tcm_cMCMC_Accept2 as
    _acceptC2 { `McmcC', `Int', `Int' } -> `()' #}

-- Do not make this function pure - it causes a segfault
{#fun pure unsafe tcm_cHH1DOpen_Local_energy as
    _locEHH1DOpenC { `McmcC', alloca- `Complex CFloat' peek* } -> `()' #}

withVectorFloat action v = V.unsafeWith action (coerce v)

-- Do not make this function pure - it causes a segfault
{#fun unsafe tcm_cDelta_well_update as
    _upDeltaWellCImpl { `Float', `RbmC', `RbmC', id `Ptr CFloat' } -> `()' #}

_upDeltaWellC :: Float -> RbmC -> RbmC -> V.Vector Float -> IO ()
_upDeltaWellC k p x rs =
  V.unsafeWith rs (\ v -> _upDeltaWellCImpl k p x (coerce v))

test :: IO ()
test = do
  p <- _newRbmC 2 4
  _printRbmC p
  _setWeightsC p (V.fromList [ 1,  0.5,  1,  1,
                              -1,   -1, -2, -1])
  _setVisibleC p (V.fromList [ 1, 1.0E-3 ])
  _setHiddenC p  (V.fromList [ 1, 1, 1, 1 ])
  _printRbmC p
  mcmc <- _newMcmcC p (V.fromList [ -1, -1 ])
  print $ _logWFC mcmc
  print $ _logQuotientWF2C mcmc 0 1
  print $ _proposeC2 mcmc 0 1
  _acceptC1 mcmc 0
  print $ _logQuotientWF2C mcmc 0 1
  _acceptC1 mcmc 0
  print $ _logWFC mcmc
  print $ _logQuotientWF2C mcmc 0 1

  -- V.unsafeWith (V.fromList [1, -1, -1, 1 :: Complex CFloat])
  --              (\v -> do mcmc <- _cMCMC_Create p v
  --                        _cMCMC_Print mcmc
  --                        print $ _cMCMC_Log_wave_function mcmc
  --              )

