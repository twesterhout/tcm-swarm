{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ViewPatterns     #-}
{-# LANGUAGE TypeFamilies     #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : Main
-- Description : Test runner
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental

module Main where


import qualified System.Random.MWC as MWC
import           Control.Lens hiding((<.>))
import           Control.Monad.Reader
import qualified Data.List            as List
import           Data.Complex
import           Data.Semigroup
import qualified Data.Vector.Storable as V
import           Numeric.LinearAlgebra((#>), (<.>))
import qualified Numeric.LinearAlgebra as LA
import           System.Exit
import           System.IO
import           System.Environment(getArgs, getProgName)
import           Foreign.Storable

import           PSO.Random
import           PSO.Swarm
import           PSO.Heisenberg
import           PSO.Energy
import           PSO.Neural

mean :: (Fractional a, Storable a) => V.Vector a -> a
mean xs = V.sum xs / fromIntegral (V.length xs)

meanVariance :: (RealFloat a, Storable a) => V.Vector a -> (a, a)
meanVariance xs
  | n > 1     = (m, sumVar m xs / fromIntegral (n - 1))
  | otherwise = (m, 0)
    where n = V.length xs
          m = mean xs
          sumVar m' = V.sum . V.map ((^2) . subtract m')


update ::
  ( m ~ ReaderT g IO
  , r ~ Float
  , χ ~ Rbm (Complex Float)
  , DeltaWell m Float χ
  , Scalable Float χ
  , Randomisable m Float
  )
  => PhaseUpdater m (SwarmGuide χ r) (BeeGuide χ r) (QMState χ) r
update = (PhaseUpdater $ deltaUpdater (1 / 2 * log 2 * 4 :: Float))

runHeisenberg ::
  ( m ~ ReaderT g IO
  , r ~ Float
  , χ ~ Rbm (Complex Float)
  , DeltaWell m Float χ
  , Scalable Float χ
  , UniformDist m Int
  , Randomisable m Float
  , Randomisable m Bool
  )
  => Int -> Int -> g -> IO ()
runHeisenberg spins count gen = do
  let h  = heisenberg1DOpen spins
      initBounds = ((-1.0E-2) :+ (-1.0E-2), 1.0E-2 :+ 1.0E-2)
      n = spins
      m = 2 * spins
      newState = QMState <$> uniformRbm n m initBounds
      func x = realPart <$> energyHH1DOpen (x ^. pos) 1000 1000
      xs = optimiseND
        newState
        update
        func
        count
        (\s -> (s^.guide.iteration == 10))
  swarms <- runReaderT xs gen
  writeEnergies2TSV "Energies.dat" (view val) swarms
  -- writeVariances2TSV "Variances.dat" swarms
  let swarm = last swarms
      (mean, v) = meanVariance . V.fromList . map (view val) $ swarm ^. bees
      min = minimum $ view val <$> swarm ^. bees
      max = maximum $ view val <$> swarm ^. bees
  -- mapM_ (print . (!!0) . (view bees)) $ swarms
  putStrLn ""
  putStrLn $ "[+] After " ++ show (swarm ^. guide . iteration)
    ++ " iterations: "
  putStrLn $ "[+] E[<H>]    = " ++ show mean
  putStrLn $ "[+] Min[<H>]    = " ++ show min
  putStrLn $ "[+] Max[<H>]    = " ++ show max
  putStrLn $ "[+] StdDev[<H>]  = " ++ show (sqrt v)
  -- putStrLn $ "[+] Best[<H>] = " ++ show (expectation h (swarm ^. guide . pos))
  -- putStrLn $ "[+] Best[Var[<H>]] = " ++ show (swarm ^. guide . val)
  putStrLn $ "[+] Actual states: " ++ show (fst . LA.eigSH $ h)


main :: IO ()
main = do
  args <- getArgs
  case args of
    [ reads -> [(spins, _)], reads -> [(count, _)], reads -> [(seed, _)]
      ] -> do g <- mkMTGen (Just seed)
              runHeisenberg spins count g
    _ -> do name <- getProgName
            hPutStrLn stderr $ "usage: " ++ name ++ " <#spins> <#bees> <seed>"
            exitFailure

