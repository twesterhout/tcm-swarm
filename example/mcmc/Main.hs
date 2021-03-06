{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ViewPatterns     #-}
{-# LANGUAGE TypeFamilies     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UnicodeSyntax #-}
--
-- |
-- Module      : Main
-- Description : Measures execution time of MC loop for 1D Heisenberg
-- Hamiltonian. Try @--help@ to get usage information.
-- Copyright   : (c) Tom Westerhout, 2018
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module Main where

import           Prelude                 hiding ( map
                                                , zipWithM
                                                )

import           Debug.Trace
import qualified System.Random.MWC             as MWC
-- import           Control.Lens hiding((<.>))
import           Control.Monad.Reader    hiding ( zipWithM )
import           Control.Monad.Primitive
import qualified Data.List                     as L
import           Data.Complex
import           Data.Semigroup
import           Data.Vector.Storable           ( Vector
                                                , (!)
                                                )
import qualified Data.Vector.Storable          as V
import qualified Data.Vector.Storable.Mutable  as MV
import           System.Exit
import           System.IO               hiding ( hGetLine )
import           System.Environment             ( getArgs
                                                , getProgName
                                                )
import           Foreign.Storable
import           Data.Text                      ( Text )
import           Data.Text.IO                   ( hGetLine )
import qualified Data.Text.IO                  as T


import           Lens.Micro
import           Lens.Micro.Extras

import           PSO.Random
import           PSO.FromPython
import           NQS.Rbm (Rbm, mkRbm)
import           NQS.Rbm.Mutable -- (sampleGradients)
import           NQS.Internal.Hamiltonian
import           NQS.Internal.Types -- (ℂ, ℝ)
import           NQS.Internal.Rbm (unsafeThawRbm)
import           NQS.Internal.LAPACK

fromPyFile :: FilePath -> IO Rbm
fromPyFile name = withFile name ReadMode toRbm
  where toRight :: Either String (Vector ℂ) -> Vector ℂ
        toRight (Right x) = x
        toRight (Left x)  = error x
        toRbm h = do
          hGetLine h
          a <- trace ("a...") $ toRight <$> readVector <$> hGetLine h
          b <- trace ("b...") $ toRight <$> readVector <$> hGetLine h
          s <- hGetLine h
          T.putStrLn s
          let !w = trace ("w...") $ toRight $ readMatrix s
          return $ mkRbm a b w

randomRbm :: Int -> Int -> (ℝ, ℝ) -> (ℝ, ℝ) -> (ℝ, ℝ) -> IO Rbm
randomRbm n m (lowV, highV) (lowH, highH) (lowW, highW) = do
  g <- mkMWCGen (Just 123)
  flip runReaderT g $ do
    visible <- uniformVector n (lowV :+ lowV, highV :+ highV)
    hidden  <- uniformVector m (lowH :+ lowH, highH :+ highH)
    weights <- uniformVector (n * m) (lowW :+ lowW, highW :+ highW)
    return $ trace ("mkRbm...") (mkRbm visible hidden weights)

numberSteps :: (Int, Int, Int) -> Int
numberSteps (low, high, step) = (high - low - 1) `div` step + 1

main = do -- NQS.Internal.LAPACK.test
  let filename = "/home/tom/src/tcm-swarm/cbits/test/input/rbm_6_6_0.in"
  rbm <- unsafeThawRbm =<< fromPyFile filename :: IO (MRbm (PrimState IO))
  -- rbm <- unsafeThawRbm =<< randomRbm 100 200 (-0.1, 0.1) (-0.1, 0.1) (-0.05, 0.05)
  let n = sizeVisible rbm
      config = defaultMCConfig & steps .~ (1000, 21000 * n, n)
                               & magnetisation .~ (Just 0)
      nParams = size rbm
      nSteps = numberSteps $ config ^. steps
  print n
  hamiltonian <- heisenberg (1, 1) (Just 5.0) (zip [0..] ([1..(n - 1)] ++ [0]))
  moments <- MV.new 4
  -- f <- newDenseVector nParams
  -- grad <- newDenseMatrix (nSteps * config ^. runs) nParams
  sampleMoments config hamiltonian rbm moments -- f grad
  e <- MV.read moments 0
  print ["Hello!", show e]
