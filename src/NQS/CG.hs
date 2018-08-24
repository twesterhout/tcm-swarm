{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_HADDOCK show-extensions #-}

module NQS.CG
  ( Operator
  , NQS.CG.cg
  , test
  ) where

import Data.Complex
import Control.Monad.Primitive

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV

import Foreign.Storable

import Lens.Micro

import NQS.Internal.Types
import NQS.Internal.BLAS


type Operator m el = -- forall m. PrimMonad m
     MDenseVector 'Direct (PrimState m) el -- ^ Input
  -> MDenseVector 'Direct (PrimState m) el -- ^ Output
  -> m ()

class ToComplex a where
  promoteToComplex :: RealOf a -> a

instance {-# OVERLAPPABLE #-} (a ~ RealOf a) => ToComplex a where
  promoteToComplex = id

instance {-# OVERLAPS #-} (Num a, a ~ RealOf (Complex a)) => ToComplex (Complex a) where
  promoteToComplex = (:+ 0)

cg :: forall v el m.
     ( PrimMonad m, v ~ MDenseVector 'Direct
     , DOTC v el el, AXPY v el, COPY v el
     , NRM2 v el (RealOf el), SCAL (RealOf el) v el
     , Storable el, Fractional el, ToComplex el, RealFloat (RealOf el)
     )
  => Int -- ^ Max number of iterations
  -> RealOf el -- ^ Tolerance
  -> Operator m el -- ^ A
  -> MDenseVector 'Direct (PrimState m) el -- ^ b
  -> MDenseVector 'Direct (PrimState m) el -- ^ Initial guess x
  -> m (Int, RealOf el)
cg !maxIter !tol operator !b !x0 = do
  r <- newTempVector n
  p <- newTempVector n
  q <- newTempVector n
  bNorm <- norm b
  -- This will probably never ever happen, but we should check anyway
  if bNorm == 0
    then fill x0 0 >> return (0, 0)
    else do
      -- r := b - max * x0
      copy b r
      operator x0 q
      axpy (-1) q r
      -- Are we done already?
      norm r >>= \rNorm -> if rNorm < tol * tol * bNorm
        then return $! (0, sqrt (rNorm / bNorm))
        else do
          -- Start the loop
          copy r p
          go bNorm x0 r p q rNorm 0
  where
    n = b ^. dim
    newTempVector size = MDenseVector size 1 <$> newVectorAligned size 64
    norm x = (\y -> y * y) <$> nrm2 x
    go !bNorm !x !r !p !q !ρ !i
      | i >= maxIter = nrm2 r >>= \rNorm -> return (i, rNorm / bNorm)
      | otherwise = do
        operator p q
        α <- (promoteToComplex ρ /) <$> dotc p q
        axpy   α  p x
        axpy (-α) q r
        norm r >>= \rNorm -> if rNorm < threshold
          then return (i, sqrt (rNorm / bNorm))
          else do copy r q
                  let ρ' = rNorm
                      β  = ρ' / ρ
                  scal β p
                  axpy 1 q p
                  go bNorm x r p q ρ' (i + 1)
      where threshold :: RealOf el
            threshold = tol * tol * bNorm

test :: IO ()
test = do
  let -- operator :: Operator Float
      operator x y = do
        aBuff <- newVectorAligned 9 64 :: IO (MV.MVector (PrimState IO) Float)
        V.copy aBuff $ V.fromList [1, 2, 3, 4, 5, 6, 7, 8, 9]
        let a = MDenseMatrix 3 3 3 aBuff :: MDenseMatrix 'Row (PrimState IO) Float
        gemv NoTranspose 1.0 a x 0.0 y
  bBuff <- V.unsafeThaw $ V.fromList [2, 2, 2 :: Float]
  xBuff <- V.unsafeThaw $ V.fromList [0, 0, 0 :: Float]
  let x = MDenseVector 3 1 xBuff :: MDenseVector 'Direct (PrimState IO) Float
  let b = MDenseVector 3 1 bBuff :: MDenseVector 'Direct (PrimState IO) Float
  print =<< (NQS.CG.cg 30 (1.0E-8 :: Float) operator b x)

