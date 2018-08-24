{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ViewPatterns     #-}
{-# LANGUAGE TypeFamilies     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE OverloadedStrings #-}
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

import           GHC.Float                      ( int2Float )

import           Lens.Micro
import           Lens.Micro.Extras
import Data.Aeson
import System.Log.FastLogger

import           PSO.Random
import           PSO.FromPython
import           NQS.Rbm (Rbm, mkRbm)
import           NQS.Rbm.Mutable -- (sampleGradients)
import           NQS.Internal.Hamiltonian
import           NQS.Internal.Rbm (unsafeThawRbm)
import           NQS.Internal.Types -- (ℂ, ℝ)
import           NQS.Internal.Sampling
import           NQS.SR

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
    return $! mkRbm visible hidden weights

numberSteps :: (Int, Int, Int) -> Int
numberSteps (low, high, step) = (high - low - 1) `div` step + 1

main = do
  -- let filename = "/home/tom/src/tcm-swarm/cbits/test/input/rbm_6_6_0.in"
  -- rbm <- unsafeThawRbm =<< fromPyFile filename :: IO (MRbm (PrimState IO))
  rbm <- unsafeThawRbm =<< randomRbm 25 50 (-0.1, 0.1) (-0.1, 0.1) (-0.05, 0.05)
  let n = sizeVisible rbm
      mcConfig = defaultMCConfig & steps .~ (1000, 1000 + 10000 * n, n)
                                 & magnetisation .~ (Just 1)
                                 & restarts .~ 20
      cgConfig = CGConfig 2000 1.0E-5
      srConfig = SRConfig 500 (Just (\i -> (10.0 * 0.9 ^ i + 0.001) :+ 0))
                              (\i -> (exp(- int2Float i / 10) + 0.03) :+ 0)
                              cgConfig mcConfig
      nParams = size rbm
      nSteps = numberSteps $ mcConfig ^. steps
  -- hamiltonian <- heisenberg (1, 1) (Just 5.0) $ (zip [0..] ([1..(n - 1)] ++ [0]))
  hamiltonian <- heisenberg (1, 1) (Just 5.5) $
    [(0,1),(0,5),(0,20),(0,4),(1,2),(1,6),(1,21),(2,3),(2,7),(2,22),(3,4),(3,8),(3,23),(4,9),(4,24),(5,6),(5,10),(5,9),(6,7),(6,11),(7,8),(7,12),(8,9),(8,13),(9,14),(10,11),(10,15),(10,14),(11,12),(11,16),(12,13),(12,17),(13,14),(13,18),(14,19),(15,16),(15,20),(15,19),(16,17),(16,21),(17,18),(17,22),(18,19),(18,23),(19,24),(20,21),(20,24),(21,22),(22,23),(23,24)]
    -- [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(6,7),(7,8),(8,6),(0,3),(3,6),(6,0),(1,4),(4,7),(7,1),(2,5),(5,8),(8,2)]
  setLogLevel Tcm_LogInfo
  withFastLogger (LogFileNoRotate "Main-Log.log" (4 * 4096)) $ \logger ->
    sr srConfig hamiltonian rbm (logger . toLogStr . (<> "\n") . Data.Aeson.encode)
  moments <- MV.new 4
  sampleMoments mcConfig hamiltonian rbm moments
  moments' <- V.freeze moments
  print moments'

