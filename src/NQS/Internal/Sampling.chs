{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE CPP #-}

module NQS.Internal.Sampling
  ( -- * Monte-Carlo sampling
    sampleMoments
  , sampleGradients
  , Tcm_Level(..)
  , setLogLevel
  ) where

import           Control.DeepSeq
import           Control.Exception              ( assert )
import           Control.Monad                  ( when
                                                , (>=>)
                                                )
import           Control.Monad.Primitive
import qualified GHC.Ptr                       as GHC
import qualified Data.Primitive.Addr           as Primitive
import qualified Data.Primitive.ByteArray      as Primitive
import           Data.Complex
import           Data.Semigroup                 ( (<>) )
import           Foreign.Storable.Complex
import           Data.Coerce
import           Data.Proxy
import           Data.Singletons
import           Data.Vector.Storable           ( Vector )
import qualified Data.Vector.Storable          as V
import           Data.Vector.Storable.Mutable   ( MVector(..)
                                                , IOVector
                                                )
import qualified Data.Vector.Storable.Mutable  as MV
import           Foreign.ForeignPtr
import           Foreign.Storable               ( Storable(..) )
import           Foreign.Ptr                    ( Ptr
                                                , castPtr
                                                , nullPtr
                                                )
import           Foreign.Marshal.Alloc          ( alloca )
import           Foreign.Marshal.Array
import           Foreign.Marshal.Unsafe
import           Foreign.Marshal.Utils   hiding ( new )
import           GHC.ForeignPtr                 ( mallocForeignPtrAlignedBytes )
import           Foreign.C.Error
import           Foreign.C.Types
import           GHC.Generics                   ( Generic )
import           System.IO.Unsafe               ( unsafePerformIO )

import           Lens.Micro
import           Lens.Micro.Extras

import           NQS.Internal.Types
import           NQS.Internal.Rbm
import           NQS.Internal.Hamiltonian

#define C2HS_IN_HOUSE
#include "nqs.h"

-- Teach c2hs about coupling of C types to Haskel types.
type Tcm_Index = {#type tcm_Index#}
{#pointer *tcm_Complex as Tcm_ComplexPtr -> Complex CFloat #}
{#pointer *tcm_Vector as Tcm_VectorPtr -> `Tcm_Vector' #}
{#pointer *tcm_Matrix as Tcm_MatrixPtr -> `Tcm_Matrix' #}
{#pointer *tcm_Hamiltonian as Tcm_HamiltonianPtr -> `Tcm_Hamiltonian' #}
{#pointer *tcm_MC_Config as Tcm_MC_ConfigPtr -> `MCConfig' #}
{#pointer *tcm_MC_Stats as Tcm_MC_StatsPtr -> `Tcm_MC_Stats' #}
{#pointer *tcm_Rbm as Tcm_RbmPtr -> `Tcm_Rbm' #}
{#enum tcm_Level as Tcm_Level
  { tcm_Level_trace as Tcm_LogTrace,
    tcm_Level_debug as Tcm_LogDebug,
    tcm_Level_info as Tcm_LogInfo,
    tcm_Level_warn as Tcm_LogWarn,
    tcm_Level_err as Tcm_LogErr,
    tcm_Level_critical as Tcm_LogCritical,
    tcm_Level_off as Tcm_LogOff }
      deriving (Eq, Show, Generic, NFData) #}

data Tcm_MC_Stats =
  Tcm_MC_Stats {-# UNPACK #-}!Tcm_Index
               {-# UNPACK #-}!ℝ

instance Storable Tcm_MC_Stats where
  sizeOf _ = {#sizeof tcm_MC_Stats#}
  {-# INLINE sizeOf #-}
  alignment _ = {#alignof tcm_MC_Stats#}
  {-# INLINE alignment #-}
  peek addr = Tcm_MC_Stats <$> {#get tcm_MC_Stats.dimension#} addr
                           <*> (coerce <$> {#get tcm_MC_Stats.variance#} addr)
  {-# INLINE peek #-}
  poke addr (Tcm_MC_Stats dimension variance) =
    {#set tcm_MC_Stats.dimension#} addr dimension *>
    {#set tcm_MC_Stats.variance#} addr (coerce variance)
  {-# INLINE poke #-}

{#fun tcm_sample_moments
  { withForeignPtr* `ForeignPtr Tcm_Rbm'
  , withForeignPtr* `ForeignPtr Tcm_Hamiltonian'
  , with* `MCConfig'
  , with* `Tcm_Vector'
  , alloca- `Tcm_MC_Stats' peek*
  } -> `()' #}

{#fun tcm_sample_gradients
  { withForeignPtr* `ForeignPtr Tcm_Rbm'
  , withForeignPtr* `ForeignPtr Tcm_Hamiltonian'
  , with* `MCConfig'
  , with* `Tcm_Vector'
  , with* `Tcm_Vector'
  , with* `Tcm_Matrix'
  , alloca- `Tcm_MC_Stats' peek*
  } -> `()' #}

{#fun unsafe tcm_set_log_level as setLogLevel { `Tcm_Level' } -> `()' #}

fromTcm_MC_Stats :: Tcm_MC_Stats -> (Int, Maybe ℝ)
fromTcm_MC_Stats (Tcm_MC_Stats dimension variance) =
  if variance == (-1.0)
    then (fromIntegral dimension, Nothing)
    else (fromIntegral dimension, Just variance)

sampleMoments
  :: PrimMonad m
  => MCConfig
  -> Hamiltonian
  -> MRbm (PrimState m)
  -> MVector (PrimState m) ℂ
  -> m (Int, Maybe ℝ)
sampleMoments config (Hamiltonian hamiltonian) (MRbm rbm) moments =
  fmap fromTcm_MC_Stats $ fmap coerce $ withTcm_Vector' moments $ \moments' ->
    unsafeIOToPrim $ tcm_sample_moments rbm hamiltonian config moments'

sampleGradients
  :: PrimMonad m
  => MCConfig
  -> Hamiltonian
  -> MRbm (PrimState m)
  -> MVector (PrimState m) ℂ
  -> MDenseVector 'Direct (PrimState m) ℂ
  -> MDenseMatrix 'Row (PrimState m) ℂ
  -> m (Int, Maybe ℝ)
sampleGradients config (Hamiltonian hamiltonian) (MRbm rbm) moments force gradients
  = fmap fromTcm_MC_Stats $ fmap coerce $ withTcm_Vector' moments $ \moments' ->
    withTcm_Vector force $ \force' -> withTcm_Matrix gradients $ \gradients' ->
      unsafeIOToPrim $ tcm_sample_gradients rbm
                                            hamiltonian
                                            config
                                            moments'
                                            force'
                                            gradients'

instance Storable MCConfig where
  sizeOf _ = {#sizeof tcm_MC_Config#}
  alignment _ = {#alignof tcm_MC_Config#}
  peek addr = MCConfig <$> peekRange addr
                       <*> peekThreads addr
                       <*> (fromIntegral <$> {#get tcm_MC_Config.runs#} addr)
                       <*> (fromIntegral <$> {#get tcm_MC_Config.flips#} addr)
                       <*> (fromIntegral <$> {#get tcm_MC_Config.restarts#} addr)
                       <*> peekMagnetisation addr
    where peekRange = {#get tcm_MC_Config.range#} >=> \p ->
            (,,) <$> (fromIntegral <$> peekElemOff p 0)
                 <*> (fromIntegral <$> peekElemOff p 1)
                 <*> (fromIntegral <$> peekElemOff p 2)
          peekThreads = {#get tcm_MC_Config.threads#} >=> \p ->
            (,,) <$> (fromIntegral <$> peekElemOff p 0)
                 <*> (fromIntegral <$> peekElemOff p 1)
                 <*> (fromIntegral <$> peekElemOff p 2)
          peekMagnetisation p =
            {#get tcm_MC_Config.has_magnetisation#} p >>= \b -> if b
              then Just <$> fromIntegral <$> {#get tcm_MC_Config.magnetisation#} p
              else return Nothing
  poke addr (MCConfig steps threads runs flips restarts magnetisation) =
    pokeRange steps addr *>
    pokeThreads threads addr *>
    {#set tcm_MC_Config.runs#} addr (fromIntegral runs) *>
    {#set tcm_MC_Config.flips#} addr (fromIntegral flips) *>
    {#set tcm_MC_Config.restarts#} addr (fromIntegral restarts) *>
    case magnetisation of
      Just m  -> {#set tcm_MC_Config.magnetisation#} addr (fromIntegral m) *>
                 {#set tcm_MC_Config.has_magnetisation#} addr True
      Nothing -> {#set tcm_MC_Config.magnetisation#} addr maxBound *>
                 {#set tcm_MC_Config.has_magnetisation#} addr False
    where pokeRange (low, high, step) = {#get tcm_MC_Config.range#} >=> \p ->
            pokeElemOff p 0 (fromIntegral low) *>
            pokeElemOff p 1 (fromIntegral high) *>
            pokeElemOff p 2 (fromIntegral step)
          pokeThreads (t1, t2, t3) = {#get tcm_MC_Config.threads#} >=> \p ->
            pokeElemOff p 0 (fromIntegral t1) *>
            pokeElemOff p 1 (fromIntegral t2) *>
            pokeElemOff p 2 (fromIntegral t3)

