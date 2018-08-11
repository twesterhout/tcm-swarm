{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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


import           Prelude hiding (map, zipWithM)

import           Debug.Trace
import qualified System.Random.MWC as MWC
-- import           Control.Lens hiding((<.>))
import           Control.Monad.Reader hiding (zipWithM)
import           Control.Monad.Primitive
import qualified Data.List            as L
import           Data.Complex
import           Data.Semigroup
import           Data.Vector.Storable (Vector, (!))
import qualified Data.Vector.Storable as V
import           System.Exit
import           System.IO hiding (hGetLine)
import           System.Environment(getArgs, getProgName)
import           Foreign.Storable
import           Data.Text (Text)
import           Data.Text.IO (hGetLine)
import qualified Data.Text.IO as T


import           Lens.Micro
import           Lens.Micro.Extras

import           PSO.Random
import           PSO.Swarm
-- import           PSO.Heisenberg
-- import           PSO.Energy
-- import           PSO.Neural
import           PSO.FromPython
import           NQS.Rbm

meanVariance :: (RealFloat a, Storable a) => V.Vector a -> (a, a)
meanVariance xs
  | n > 1     = (m, sumVar m xs / fromIntegral (n - 1))
  | otherwise = (m, 0)
    where n = V.length xs
          m = V.sum xs / fromIntegral (V.length xs)
          sumVar m' = V.sum . V.map ((^2) . subtract m')

type R = Float
type C = Complex Float
type X = Rbm C

fromPyFile :: FilePath -> IO X
fromPyFile name = withFile name ReadMode toRbm
  where toRight :: Either String (V.Vector (Complex Float)) -> V.Vector (Complex Float)
        toRight (Right x) = x
        toRight (Left x)  = error x
        toRbm h = do
          a <- trace ("a...") $ toRight <$> readVector <$> hGetLine h
          b <- trace ("b...") $ toRight <$> readVector <$> hGetLine h
          s <- hGetLine h
          T.putStrLn s
          let !w = trace ("w...") $ toRight $ readMatrix s
          return $ mkRbm (a, b, w)

randomRbm ::
  ( UniformDist m R
  )
  => Int
  -> Int
  -> (R, R)
  -> (R, R)
  -> (R, R)
  -> m X
randomRbm n m (lowV, highV) (lowH, highH) (lowW, highW) =
  do
    visible <- uniformVector n (lowV :+ lowV, highV :+ highV)
    hidden <- uniformVector m (lowH :+ lowH, highH :+ highH)
    weights <- uniformVector (n * m) (lowW :+ lowW, highW :+ highW)
    return $ trace ("mkRbm...") (mkRbm (visible, hidden, weights))

rbms :: (UniformDist m R) => m [X]
rbms = trace ("Creating rbms...") $ replicateM 20 (randomRbm 10 20 (-0.1, 0.1) (-0.1, 0.1) (-0.1, 0.1))

rbms' :: IO [X]
rbms' = Prelude.mapM fromPyFile $
  [ "test.txt"
  -- "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-1.23981661958_energy"
  -- "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-1.89541865304_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_0.0887681979723_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_1.40938434022_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-3.61205064542_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-4.1917579599_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-5.41817224673_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-5.74064288804_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-6.33997058402_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-6.48266135073_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-7.29704001394_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-7.4024108663_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-7.83407453451_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-8.17005224883_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-8.18318151693_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-8.75410430005_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-9.04188789775_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-9.34302369964_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_-9.5917604763_energy"
  -- , "/home/tom/src/tcm-swarm/10_spins/Weights_10_spins_2_density_2.01304877555_energy"
  ]

{-
  [ "/home/tom/src/tcm-swarm/8_spins/input_singleflip_0.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_1.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_2.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_3.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_4.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_5.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_6.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_7.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_8.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_9.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_10.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_11.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_12.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_13.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_14.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_15.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_16.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_17.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_18.txt"
  , "/home/tom/src/tcm-swarm/8_spins/input_singleflip_19.txt"
  ]
-}


function :: QMState X -> ReaderT g IO (Vector C)
function x = lift $
  sampleMoments (x ^. pos) Heisenberg1DPeriodic 4 (1000, 7000 + 1000, 7) (Just 0) 4

instance {-# OVERLAPS #-} Ord (Vector C) where
  a <= b = let k x = realPart (x ! 3) / (realPart (x ! 1))^2
            in k a <= k b

process :: Swarm m (SwarmGuide X (Vector C)) (BeeGuide X (Vector C)) (QMState X) (Vector C)
        -> IO ()
process swarm = do
  putStrLn $ "At iteration #" <> show (swarm ^. guide . iteration)
  withFile "Energies.dat" AppendMode $ \h ->
    do
      let xs = view bees swarm
      hPutStrLn h (L.concat $ L.intersperse "\t" $ show . realPart . (! 0) . view val <$> xs)
  withFile "Variances.dat" AppendMode $ \h ->
    do
      let xs = view bees swarm
      hPutStrLn h (L.concat $ L.intersperse "\t" $ show . realPart . (! 1) . view val <$> xs)
  withFile "Kurtosis.dat" AppendMode $ \h ->
    do
      let xs = view bees swarm
          k x = realPart (x ! 3) / (realPart (x ! 1))^2
      hPutStrLn h (L.concat $ L.intersperse "\t" $ show . k . view val <$> xs)

instance Scalable Float (Rbm C) where
  scale λ x = map (* (λ :+ 0)) x

instance DeltaWell m R R => DeltaWell m R X where
  upDeltaWell κ p x = zipWithM (upDeltaWell κ) p x

update ::
  ( m ~ ReaderT g IO
  , Randomisable m Float
  )
  => PhaseUpdater m (SwarmGuide X r) (BeeGuide X r) (QMState X) r
update = PhaseUpdater $ deltaUpdater (1.9 * log 2 :: R)

runHeisenbergFromList ::
  ( m ~ ReaderT g IO
  -- , r ~ MeanVar Float
  -- , χ ~ Rbm (Complex Float)
  -- , DeltaWell m Float χ
  -- , Scalable Float χ
  , UniformDist m Int
  , Randomisable m Float
  , Randomisable m Bool
  )
  => g -> IO ()
runHeisenbergFromList gen = runReaderT go gen
  where go =
          do
            -- states <- fmap QMState <$> rbms 
            states <- fmap QMState <$> lift rbms'
            optimiseNDFromList
                  states
                  update
                  function
                  (\s -> (s^.guide.iteration == 100))
                  (lift . process)
            return ()

{-
runHeisenberg ::
  ( m ~ ReaderT g IO
  -- , r ~ MeanVar Float
  -- , χ ~ Rbm (Complex Float)
  -- , DeltaWell m Float χ
  -- , Scalable Float χ
  , UniformDist m Int
  , Randomisable m Float
  , Randomisable m Bool
  )
  => Int -> Int -> g -> IO ()
runHeisenberg spins count gen = do
  let h  = heisenberg1DOpen spins
      initBounds = ((-2.0E-1) :+ (-2.0E-1), 2.0E-1 :+ 2.0E-1)
      n = spins
      newState = QMState <$> uniformRbm n (2 * spins) initBounds
      func x = uncurry MV <$> energyHH1DOpenMKLC (x ^. pos) 100 10000
      xs = optimiseND
        newState
        update
        func
        count
        (\s -> (s^.guide.iteration == 100))
  swarms <- runReaderT xs gen
  writeEnergies2TSV "Energies.dat"  (view (val . mean)) swarms
  writeEnergies2TSV "Variances.dat" (view (val . var)) swarms
  -- writeVariances2TSV "Variances.dat" swarms
  let swarm = last swarms
      (m, v) = meanVariance
                . V.fromList
                . map (view (val . mean))
                $ swarm ^. bees
      min = minimum $ view mean <$> view val <$> swarm ^. bees
      max = maximum $ view mean <$> view val <$> swarm ^. bees
  -- mapM_ (print . (!!0) . (view bees)) $ swarms
  putStrLn ""
  putStrLn $ "[+] After " ++ show (swarm ^. guide . iteration)
    ++ " iterations: "
  putStrLn $ "[+] E[<H>]         = " ++ show m
  putStrLn $ "[+] Min[<H>]       = " ++ show min
  putStrLn $ "[+] Max[<H>]       = " ++ show max
  putStrLn $ "[+] StdDev[<H>]    = " ++ show (sqrt v)
  putStrLn $ "[+] Best[<H>]      = " ++ show (swarm ^. guide . val . mean)
  putStrLn $ "[+] Best[Var[<H>]] = " ++ show (swarm ^. guide . val . var)
  putStrLn $ "[+] Actual states:   " ++ show (fst . LA.eigSH $ h)
-}

main :: IO ()
main = do
  args <- getArgs
  case args of
    [ reads -> [(spins :: Int, _)], reads -> [(count :: Int, _)], reads -> [(seed, _)]
      ] -> do g <- mkMWCGen (Just seed)
              runHeisenbergFromList g
    _ -> do name <- getProgName
            hPutStrLn stderr $ "usage: " ++ name ++ " <#spins> <#bees> <seed>"
            exitFailure

