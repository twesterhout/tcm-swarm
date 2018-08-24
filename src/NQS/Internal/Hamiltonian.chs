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

module NQS.Internal.Hamiltonian
  ( -- * Low-level interface to Hamiltonians
    Tcm_Hamiltonian(..)
  , Hamiltonian(..)
  , HamiltonianType(..)
  , heisenberg
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


#define C2HS_IN_HOUSE
#include "nqs.h"

{#enum tcm_HamiltonianType as HamiltonianType
  { TCM_SPIN_HEISENBERG as TCM_SPIN_HEISENBERG,
    TCM_SPIN_ANISOTROPIC_HEISENBERG as TCM_SPIN_ANISOTROPIC_HEISENBERG,
    TCM_SPIN_DZYALOSHINSKII_MORIYA as TCM_SPIN_DZYALOSHINSKII_MORIYA }
      deriving (Eq, Show, Generic, NFData) #}

{#pointer *tcm_Hamiltonian as Tcm_HamiltonianPtr -> `Tcm_Hamiltonian' #}

type Tcm_Index = {#type tcm_Index#}

-- | Represents a Hamiltonian.
newtype Hamiltonian = Hamiltonian (ForeignPtr Tcm_Hamiltonian)

-- | Low-level polymorphic representation of Hamiltonians.
--
-- It should be binary compatible (through 'Storable' interface) with
-- @tcm_Hamiltonian@ C struct.
data Tcm_Hamiltonian =
  Tcm_Hamiltonian !(Ptr ())        -- ^ Pointer to the C++ class.
                  !HamiltonianType -- ^ Type of the C++ class.
  deriving (Show, Generic, NFData)

instance Storable Tcm_Hamiltonian where
  sizeOf _ = {#sizeof tcm_Hamiltonian#}
  alignment _ = {#alignof tcm_Hamiltonian#}
  peek ptr =
    Tcm_Hamiltonian <$> {#get tcm_Hamiltonian.payload#} ptr
                    <*> (toEnum . fromIntegral <$> {#get tcm_Hamiltonian.dtype#} ptr)
  poke ptr (Tcm_Hamiltonian payload dtype) =
    {#set tcm_Hamiltonian.payload#} ptr payload *>
    {#set tcm_Hamiltonian.dtype#} ptr (fromIntegral . fromEnum $ dtype)

-- | Represents an edge in a graph.
--
-- This type is provided solely for binary compatibility with C's @int [2]@ and
-- C++' @std::array<int, 2>@.
data Edge = Edge {-# UNPACK #-}!Tcm_Index {-# UNPACK #-}!Tcm_Index

instance Storable Edge where
  sizeOf _    = 2 * sizeOf (undefined :: Tcm_Index)
  alignment _ = alignment (undefined :: Tcm_Index)
  peek p = Edge <$> peekElemOff (castPtr p) 0 <*> peekElemOff (castPtr p) 1
  poke p (Edge i j) = pokeElemOff (castPtr p) 0 i *> pokeElemOff (castPtr p) 1 j

{#fun unsafe tcm_Heisenberg_create
  { id `Ptr Tcm_Hamiltonian', id `Ptr (Ptr Tcm_Index)', `Int', `Int', `Int', id `Ptr CFloat' } -> `()' #}

foreign import ccall unsafe "nqs.h &tcm_Heisenberg_destroy"
  tcm_Heisenberg_destroy :: FinalizerPtr Tcm_Hamiltonian

-- | Constructs an isotropic (with coupling of 1) Heisenberg Hamiltonian on a
-- lattice described by a list of edges.
--
-- For example, you can construct Heisenberg Hamiltonian with periodic boundary
-- conditions on 1D spin chain of 5 electrons using
--
-- > heisenberg [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
--
-- /Note/ that indexing starts from 0!
heisenberg :: (Int, Int) -> Maybe Float -> [(Int, Int)] -> IO Hamiltonian
heisenberg (t1, t2) cutoff edges =
  maybeWith with (coerce <$> cutoff) $ \cutoffPtr ->
  withArrayLen (toEdge <$> edges) $ \n p -> do
    h <- mallocForeignPtr
    withForeignPtr h $ \hPtr -> tcm_Heisenberg_create hPtr (castPtr p) n t1 t2 cutoffPtr
    addForeignPtrFinalizer tcm_Heisenberg_destroy h
    return $! Hamiltonian h
  where
    toEdge !(i, j) = Edge (fromIntegral i) (fromIntegral j)
