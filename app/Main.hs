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
import PSO.Swarm

prettyGuide :: (Show p, Show r) => BeeGuide p r -> String
prettyGuide x = "f_min = f(" ++ show (x^.pos) ++ ") =\t"
  ++ show (x^.val)

prettyBee :: (Show p, Show r) => Bee p r (BeeGuide p r) -> String
prettyBee x = "(*) x = " ++ show (x^.pos) ++ "; "
    ++ "v = " ++ show (x^.vel) ++ "; " ++ prettyGuide (x^.guide)

prettySwarm :: (Show p, Show r)
            => Swarm m gen (BeeGuide p r) (BeeGuide p r) p r -> String
prettySwarm xs = "[" ++ show (xs^.iteration) ++ "] " ++ replicate 30 '=' ++ "\n"
    ++ L.concat (L.intersperse "\n" (map prettyBee (xs^.bees))) ++ "\n"
    ++ "\t==> " ++ prettyGuide (xs^.stats)

compactSwarm :: (Show p, Show r, s ~ g, g ~ BeeGuide p r)
            => Swarm m gen s g p r -> String
compactSwarm xs = "[" ++ show (xs^.iteration) ++ "]\t"
  ++ prettyGuide (xs^.stats)

-- run1D :: IO ()
-- run1D = do
--   let xs = optimise1D
--         (-0.12 :: Float, 1.7, 1.8)  -- (inertia, local, global) params
--         (-5, 5)                     -- 1D (low, high) bounds
--         (\x -> (x - 4) * x)         -- 1D function to minimise
--         20                          -- number of bees in the swarm
--         (\s -> s^.iteration == 10) -- number of iterations
--   g <- mkMWCGen (Just 10)           -- create a RNG with seed 10
--   runReaderT xs g >>= mapM_ (putStrLn . prettySwarm)

wpg :: (Float, Float, Float)
wpg = (0.7298 :: Float, 1.49618, 1.49618)

banana :: V.Vector Float -> Float
banana = (\x -> 100 * ((x!1) - (x!0))**2 + (1 - (x!0))**2)

runND :: IO ()
runND = do
  let xs = optimiseND
        (standardUpdater wpg)
        [(-2, 1), (-2, 1)]
        [(-5, 5), (-5, 5)]
        (\x -> 100 * ((x!1) - (x!0))**2 + (1 - (x!0))**2 :: Float)
        30
        (\s -> s^.iteration == 100) -- number of iterations
  g <- mkMWCGen Nothing           -- create a RNG with seed 10
  runReaderT xs g >>= mapM_ (putStrLn . compactSwarm)

-- run1DC :: IO ()
-- run1DC = do
--   let xs = optimise1D
--         ( 0.7298 :: Complex Float
--         , 1.49618 :: Complex Float
--         , 1.49618 :: Complex Float
--         )
--         ((-5) :+ (-5), 5 :+ 5)
--         (\(x :+ y) -> (x - 1)^^2 + (y - 2)^^2)
--         30
--         (\s -> s^.iteration == 10) -- number of iterations
--   g <- mkMWCGen (Just 10)           -- create a RNG with seed 10
--   runReaderT xs g >>= mapM_ (putStrLn . prettySwarm)

runNDC :: IO ()
runNDC = do
  let f (x :+ y) = (x - 1)^^2 + (y - 2)^^2
      xs = optimiseND
        (standardUpdater ( 0.7298  :: Complex Float
                         , 1.49618 :: Complex Float
                         , 1.49618 :: Complex Float))
        [ ((-5) :+ (-5), 5 :+ 5)
        , ((-5) :+ (-5), 5 :+ 5)
        ]
        [ ((-5) :+ (-5), 5 :+ 5)
        , ((-5) :+ (-5), 5 :+ 5)
        ]
        (\x -> f (x!0) + f (x!1) :: Float)
        30
        (\s -> s^.stats.val < 1.0E-5)
  g <- mkMWCGen Nothing           -- create a RNG with seed 10
  runReaderT xs g >>= mapM_ (putStrLn . compactSwarm)



-- runHeisenberg :: IO ()
-- runHeisenberg = do
--   let h = heisenberg1D4Open
--       expect m x = let x' = LA.normalize x
--                     in realPart $ LA.dot x' (m LA.#> x')
--       f x = ((expect (h LA.<> h) x) - (expect h x)^^2)^^2
--       xs = optimiseND
--         ( 0.7298
--         , 1.49618
--         , 1.49618
--         )
--         (replicate (2^4) ((-1) :+ (-1), 1 :+ 1))
--         f
--         50
--         (\s -> s^.guide.val < 1.0E-8)
--   g <- mkMWCGen (Just 129)
--   swarms <- runReaderT xs g
--   writeEnergies2TSV "Energies.dat" (\x -> expect h (x^.pos)) swarms
--   print $ expect h ((last swarms)^.guide.pos)







main :: IO ()
main = do
  putStrLn "Bye!"
