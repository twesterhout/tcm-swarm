{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_HADDOCK show-extensions #-}

-- {-# LANGUAGE OverlappingInstances   #-}
-- |
-- Module      : Energy
-- Description : Particle Swarm Optimisation
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.Energy
  ( writeEnergies2TSV
  , writeVariances2TSV
  ) where

import Control.Lens
import Data.Monoid
import Data.List(intersperse)
import qualified Data.List as List
import qualified Numeric.LinearAlgebra as LA
import qualified Data.ByteString.Lazy as LB
import Data.ByteString.Builder

import PSO.Swarm


class EncodableAsDecimal a where
  encodeAsDecimal :: a -> Builder

instance EncodableAsDecimal Float where
  encodeAsDecimal = floatDec

instance EncodableAsDecimal Double where
  encodeAsDecimal = doubleDec


energies2TSV :: (EncodableAsDecimal t)
             => (Bee α β r -> t) -> [Swarm m γ β α r] -> LB.ByteString
energies2TSV func =
  let encodeSwarm = mconcat . intersperse (charUtf8 '\t')
        . map (encodeAsDecimal . func) . view bees
   in toLazyByteString . mconcat . intersperse (charUtf8 '\n')
        . map encodeSwarm

variances2TSV :: (EncodableAsDecimal r, HasVar γ r)
              => [Swarm m γ β α r] -> LB.ByteString
variances2TSV = toLazyByteString . mconcat . intersperse (charUtf8 '\n')
                . map (encodeAsDecimal . view (guide . var))

writeEnergies2TSV :: (EncodableAsDecimal t)
                  => FilePath -> (Bee α β r -> t) -> [Swarm m γ β α r] -> IO ()
writeEnergies2TSV file func = LB.writeFile file . energies2TSV func

writeVariances2TSV :: (EncodableAsDecimal r, HasVar γ r)
                   => FilePath -> [Swarm m γ β α r] -> IO ()
writeVariances2TSV file = LB.writeFile file . variances2TSV
