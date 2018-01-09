{-# LANGUAGE FlexibleContexts #-}
-- |
-- Module      : PSO.NeuralSpec
-- Description : Tests for PSO.Neural
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental

module PSO.NeuralSpec where

import           Control.Monad
import           Control.Monad.ST
import           Control.Monad.Reader
import           Data.Complex
import qualified Data.Vector.Storable as V
import           Foreign.Storable
import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Devel as LA.Devel

import           Test.Hspec

import           PSO.Random
import           PSO.Neural



main :: IO ()
main = hspec spec

roundTo :: (RealFrac a) => Int -> a -> a
roundTo n x = (fromInteger . round $ x * (10^n)) / (10.0^^n)

roundToC :: (RealFrac a) => Int -> Complex a -> Complex a
roundToC n (x :+ y) = (roundTo n x) :+ (roundTo n y)


class AlmostEq a where
  eq' :: a -> a -> Bool

instance AlmostEq Float where
  eq' x y = let epsilon = 1.0E-5
             in abs (x - y) <= epsilon * max (abs x) (abs y)

instance AlmostEq a => AlmostEq (Complex a) where
  eq' (xr :+ xi) (yr :+ yi) = xr `eq'` yr && xi `eq'` yi

instance (Storable a, AlmostEq a) => AlmostEq (V.Vector a) where
  eq' x y = V.and $ V.zipWith eq' x y


toW n m w = LA.Devel.matrixFromVector LA.Devel.ColumnMajor m n w

mkTheta a b w' σ = b + (toW (V.length a) (V.length b) w') LA.#> σ

logWF' :: (Storable a, RealFloat a, Num (V.Vector (Complex a)), LA.Numeric (Complex a))
       => V.Vector (Complex a)
       -> V.Vector (Complex a)
       -> V.Vector (Complex a)
       -> V.Vector (Complex a)
       -> Complex a
logWF' a b w' σ =
  let n = V.length a
      m = V.length b
      θ = mkTheta a b w' σ
   in V.sum (V.zipWith (*) a σ) + V.sum (V.map (log . cosh) θ)

logQuotient1' ::
     (Storable a, RealFloat a, Num (V.Vector (Complex a)), LA.Numeric (Complex a))
  => V.Vector (Complex a)
  -> V.Vector (Complex a)
  -> V.Vector (Complex a)
  -> V.Vector (Complex a)
  -> Int -> Complex a
logQuotient1' a b w' σ flip =
  let σ' = σ V.// [(flip, (-1) * (σ V.! flip))]
   in logWF' a b w' σ' - logWF' a b w' σ

logQuotient2' ::
     (Storable a, RealFloat a, Num (V.Vector (Complex a)), LA.Numeric (Complex a))
  => V.Vector (Complex a)
  -> V.Vector (Complex a)
  -> V.Vector (Complex a)
  -> V.Vector (Complex a)
  -> Int -> Int -> Complex a
logQuotient2' a b w' σ flip1 flip2 =
  let σ' = σ V.// [ (flip1, (-1) * (σ V.! flip1))
                  , (flip2, (-1) * (σ V.! flip2)) ]
   in logWF' a b w' σ' - logWF' a b w' σ

locEnergyHH1DOpen' ::
     (Storable α, RealFloat α, Eq α, Num (V.Vector (Complex α)), LA.Numeric (Complex α))
  => V.Vector (Complex α)
  -> V.Vector (Complex α)
  -> V.Vector (Complex α)
  -> V.Vector (Complex α)
  -> Complex α
locEnergyHH1DOpen' a b w' σ =
  let zipper i x y
        | x == y    = 1
        | otherwise = -1 + 2 * exp (logQuotient2' a b w' σ i (i + 1))
   in V.sum $ V.izipWith zipper σ (V.tail σ)

a1 = V.fromList [   3.7e-01  :+   1.5e-01
                ,   6.5e-01  :+   3.0e-01
                , (-2.8e-01) :+ (-3.0e-01)
                ] :: V.Vector (Complex Float)

ra1 :: UniformDist m (Complex Float) => m (V.Vector (Complex Float))
ra1 = uniformVector 10 ((-1.0) :+ (-1.0), 1.0 :+ 1.0)

b1 = V.fromList [   6.0e-01  :+   8.4e-02
                , (-2.6e-01) :+   9.2e-01
                ,   5.5e-01  :+ (-8.2e-02)
                ,   4.3e-01  :+   3.7e-01
                ,   9.3e-01  :+   5.1e-01
                ,   6.0e-02  :+   8.9e-01
                ] :: V.Vector (Complex Float)

rb1 :: UniformDist m (Complex Float) => m (V.Vector (Complex Float))
rb1 = uniformVector 20 ((-1.0) :+ (-1.0), 1.0 :+ 1.0)

w1 = V.fromList $
      [   0.22 :+ (-0.93),   (-0.16) :+ (-0.47), (-0.57) :+ (-0.83),   (-0.16) :+ 0.86,    (-0.44) :+ 0.16,       0.43 :+ 0.96
      , (-3.6e-2) :+ 0.36, (-1.5e-2) :+ (-0.14),    (-0.15) :+ 0.57,   0.79 :+ (-0.26),     0.95 :+ (-0.6),       0.56 :+ 0.98
      ,      0.82 :+ 0.34,      0.85 :+ (-0.35),    0.64 :+ (-0.24), (-0.73) :+ (-0.8), (-0.34) :+ (-0.24), (-0.94) :+ (-0.46)
      ] :: V.Vector (Complex Float)

rw1 :: UniformDist m (Complex Float) => m (V.Vector (Complex Float))
rw1 = uniformVector (10 * 20) ((-1.0) :+ (-1.0), 1.0 :+ 1.0)

σ1  = V.fromList [ -1, -1, -1 ] :: V.Vector (Complex Float)
σ2  = V.fromList [ 1,   1,  1 ] :: V.Vector (Complex Float)
σ3  = V.fromList [ 1,  -1,  1 ] :: V.Vector (Complex Float)

rσ1 :: Randomisable m Bool => m (V.Vector (Complex Float))
rσ1 = randomSpin 10

rbm1  = runST $ mkRbm a1 b1 w1

mcmc1 = runST $ newMcmc rbm1 σ1 >>= unsafeFreezeMcmc
mcmc2 = runST $ newMcmc rbm1 σ2 >>= unsafeFreezeMcmc
mcmc3 = runST $ newMcmc rbm1 σ3 >>= unsafeFreezeMcmc


spec :: Spec
spec = do
  describe "mkRbm" $ do
    it "Constructs a new RBM given a, b, and w" $
      debugPrintRbm rbm1
  describe "newMcmc" $ do
    it "Constructs a new MCMC given RBM and σ" $
      do debugPrintMcmc mcmc1
         print $ (b1 + (toW 3 6 w1) LA.#> σ1)
  describe "logWF" $ do
    it "1) Calculates ln(ψ(S)) " $
      do putStrLn $ "logWF:  " ++ show (logWF mcmc1)
         putStrLn $ "logWF': " ++ show (logWF' a1 b1 w1 σ1)
    it "2) Calculates ln(ψ(S)) " $
      do putStrLn $ "logWF:  " ++ show (logWF mcmc2)
         putStrLn $ "logWF': " ++ show (logWF' a1 b1 w1 σ2)
    it "3) Calculates ln(ψ(S)) " $
      do putStrLn $ "logWF:  " ++ show (logWF mcmc3)
         putStrLn $ "logWF': " ++ show (logWF' a1 b1 w1 σ3)
  describe "logQuotient1" $ do
    it "1) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient1:  " ++ show (logQuotient1 mcmc1 0)
         putStrLn $ "logQuotient1': " ++ show (logQuotient1' a1 b1 w1 σ1 0)
    it "2) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient1:  " ++ show (logQuotient1 mcmc1 1)
         putStrLn $ "logQuotient1': " ++ show (logQuotient1' a1 b1 w1 σ1 1)
    it "3) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient1:  " ++ show (logQuotient1 mcmc1 2)
         putStrLn $ "logQuotient1': " ++ show (logQuotient1' a1 b1 w1 σ1 2)
    it "4) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient1:  " ++ show (logQuotient1 mcmc2 0)
         putStrLn $ "logQuotient1': " ++ show (logQuotient1' a1 b1 w1 σ2 0)
    it "5) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient1:  " ++ show (logQuotient1 mcmc2 2)
         putStrLn $ "logQuotient1': " ++ show (logQuotient1' a1 b1 w1 σ2 2)
  describe "logQuotient2" $ do
    it "1) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient2:  " ++ show (logQuotient2 mcmc1 0 1)
         putStrLn $ "logQuotient2': " ++ show (logQuotient2' a1 b1 w1 σ1 0 1)
    it "2) Calculates ln(ψ(S')/ψ(S))" $
      do putStrLn $ "logQuotient2:  " ++ show (logQuotient2 mcmc1 0 2)
         putStrLn $ "logQuotient2': " ++ show (logQuotient2' a1 b1 w1 σ1 0 2)
  describe "locEnergyHH1DOpen'" $ do
    it "1) Calculates Eloc(ψ)" $
      do putStrLn $ "locEnergyHH1DOpen:  " ++ show (locEnergyHH1DOpen mcmc3)
         putStrLn $ "locEnergyHH1DOpen': " ++ show (locEnergyHH1DOpen' a1 b1 w1 σ3)
  describe "Random!" $ do
    it "1) Calculates Eloc(ψ)" $
      do let job = do
               a <- ra1
               b <- rb1
               w <- rw1
               σ <- rσ1
               rbm <- mkRbm a b w
               mcmc <- newMcmc rbm σ >>= unsafeFreezeMcmc
               lift $ putStrLn
                    $ "locEnergyHH1DOpen:  " ++ show (locEnergyHH1DOpen mcmc)
               lift $ putStrLn
                    $ "locEnergyHH1DOpen': " ++ show (locEnergyHH1DOpen' a b w σ)
         runReaderT job =<< mkMWCGen (Just 135)
         runReaderT job =<< mkMWCGen (Just 136)
         runReaderT job =<< mkMWCGen (Just 137)
