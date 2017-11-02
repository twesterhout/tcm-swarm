
-- |
-- Module      : Main
-- Description : Test runner
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental

module Main where

import Swarm
import System.Random
import Control.Lens
import Control.Monad.State
import qualified Data.List as L

prettyGuide :: (Show p, Show r) => PSOGuide p r -> String
prettyGuide x = "f_min = f(" ++ show (x^.pos) ++ ") = " ++ show (x^.val)

prettyBee :: (Show p, Show r) => Bee p r -> String
prettyBee x = "(*) x = " ++ show (x^.pos) ++ "; "
    ++ "v = " ++ show (x^.vel) ++ "; " ++ prettyGuide (x^.guide)

prettySwarm :: (Show p, Show r) => Swarm g p r -> String
prettySwarm xs = "[" ++ show (xs^.iteration) ++ "] " ++ replicate 30 '=' ++ "\n"
    ++ L.concat (L.intersperse "\n" (map prettyBee (xs^.bees))) ++ "\n"
    ++ "\t==> " ++ prettyGuide (xs^.guide)


main :: IO ()
main = do
    let xs = optimise1D (-0.12, 1.7, 1.8)     -- (inertia, local, global) params
                        (-5, 5)               -- 1D (low, high) bounds
                        (\x -> (x - 4) * x)   -- 1D function to minimise
                        20                    -- number of bees in the swarm
    mapM_ (putStrLn . prettySwarm) -- ignore this
        . take 5                   -- number of iterations
        . fst                      -- ignore this
        . runState xs              -- ignore this
        $ mkStdGen 11              -- seed for random numbers

