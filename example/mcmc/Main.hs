-- |
-- Module      : Main
-- Description : Measures execution time of MC loop for 1D Heisenberg
-- Hamiltonian. Try @--help@ to get usage information.
-- Copyright   : (c) Tom Westerhout, 2018
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module Main where

import Control.Monad.Par
import Control.Monad.Par.Combinator
import Control.Monad.Par.IO as ParIO
-- import Control.Exception (evaluate)
import Control.Monad.Reader
import Data.Complex

-- import Data.Semigroup ((<>))
import Data.Word
import System.IO

import PSO.Neural
import PSO.Random

main = do
  let n = 8
      m = 16
      norm = 1.0E-2 :: Float
  rbmGen <- mkMTGen (Just 180 :: Maybe Word32)
  rbm <-
    runReaderT (uniformRbm n m ((-norm) :+ (-norm), norm :+ norm)) rbmGen 
      :: IO (Rbm (Complex Float))
  answer <-
    ParIO.runParIO $
    parMapM (\x -> liftIO $ energyHH1DOpenMKLC x 1024 2048) $
    replicate 500 rbm
  forM_
    answer
    (\(Measurement mean var) -> putStrLn $ show mean ++ "\t" ++ show var)
