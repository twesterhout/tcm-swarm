{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies     #-}
-- |
-- Module      : Main
-- Description : Test runner
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental

module Main where


import qualified System.Random.MWC as MWC
import           Control.Lens
import           Control.Monad.Reader
import           Control.Monad.State
import qualified Data.List            as L
import Data.Complex
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import Data.Vector.Storable((!))

import PSO.Random
import PSO.Energy
import PSO.Swarm


class TestFunction f where
  function :: f -> V.Vector Float -> Float
  initBounds :: f -> (Float, Float)
  dim :: f -> Int


data RosenbrockFn = RosenbrockFn

instance TestFunction RosenbrockFn where
  function _ x =
    let p1 = (100.0 *) . V.sum . V.map (^^2)
                $ V.zipWith (-) (V.tail x) (V.map (^^2) $ V.init x)
        p2 = V.sum . V.map (^^2) . V.map (1.0 -) $ V.init x
    in p1 + p2
  initBounds _ = (15.0, 30.0)
  dim _ = 30

unpack :: (Float, Float) -> Int -> (V.Vector Float, V.Vector Float)
unpack (x, y) d = (f x, f y)
  where f = V.fromList . replicate d

wpg :: (Float, Float, Float)
wpg = (0.7298 :: Float, 1.49618, 1.49618)

cmUpdater :: (Num χ, RandomScalable m χ, VectorSpace χ Float)
        => PhaseUpdater m (SwarmGuide χ r) (BeeGuide χ r) (CMState χ) r
cmUpdater = standardUpdater wpg

qmUpdater :: (RandomScalable m χ, Randomisable m Float, χ ~ V.Vector Float)
        => PhaseUpdater m (SwarmGuide χ r) (BeeGuide χ r) (QMState χ) r
qmUpdater = (PhaseUpdater $ deltaWellUpdater (2 * log 2 * 0.7 :: Float))

cmRunND :: IO ()
cmRunND = do
  let xs = optimiseND
            (mkCMState (unpack (initBounds RosenbrockFn) (dim RosenbrockFn)))
            cmUpdater
            (function RosenbrockFn)
            20
            (\s -> (s^.guide.val) < 1.0
                   || (s^.guide.iteration == 10000))
  gen <- mkMWCGen (Just 123)
  swarms <- runReaderT xs gen
  writeEnergies2TSV "Function.dat" (view val) swarms
  let swarm = last swarms
  -- mapM_ (print . (!!0) . (view bees)) $ swarms
  putStrLn ""
  putStrLn $ "[+] Best[f] = " ++ show (swarm ^. guide . val)

qmRunND :: IO ()
qmRunND = do
  let xs = optimiseND
            (mkQMState (unpack (initBounds RosenbrockFn) (dim RosenbrockFn)))
            qmUpdater
            (function RosenbrockFn)
            20
            (\s -> (s^.guide.val) < 1.0
                   || (s^.guide.iteration == 5000))
  gen <- mkMWCGen (Just 123)
  swarms <- runReaderT xs gen
  writeEnergies2TSV "Function.dat" (view val) swarms
  let swarm = last swarms
  -- mapM_ (print . (!!0) . (view bees)) $ swarms
  putStrLn ""
  putStrLn $ "[+] Best[f] = " ++ show (swarm ^. guide . val)

main :: IO ()
main = qmRunND
