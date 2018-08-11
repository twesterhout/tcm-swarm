{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : PSO.Heisenberg
-- Description : Heisenberg Hamiltonian
-- Copyright   : (c) Tom Westerhout, 2017
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.Heisenberg
  ( -- * Introduction
    --
    -- | Let's start with a 1D chain of \(N\) spin-1/2 particles. Each of them
    -- is described by a state \(|\alpha_i\rangle\in\mathbb{C}^2\). Hilbert
    -- space of the whole system is then \(\bigotimes_{i=0}^{N-1}\mathbb{C}^2\).
    --
    -- We use the following Hamiltonian to describe this system:
    -- \[
    -- \hat{\mathcal{H}} = \sum_{i=0}^{N-2}
    --    \hat{\boldsymbol\sigma}_i \cdot \hat{\boldsymbol\sigma}_{i+1}
    --    = \sum_{i=0}^{N-2}
    --          \hat{\sigma}_i^{(x)}\hat{\sigma}_{i+1}^{(x)}
    --        + \hat{\sigma}_i^{(y)}\hat{\sigma}_{i+1}^{(y)}
    --        + \hat{\sigma}_i^{(z)}\hat{\sigma}_{i+1}^{(z)} \;.
    -- \]
    -- Only the nearest neighbours are connected. The boundary conditions used
    -- are called /open/ because the zeroth and N-1st spins don't interact.
    one2D
  , pauliX
  , pauliY
  , pauliZ
  , heisenberg1DOpen
    -- * Quantities of interest
    --
    -- | The quantity we're actually interested in is
    -- \[
    -- \frac{\langle\sigma| \hat{\mathcal{H}} |\psi\rangle}{%
    --   \langle\sigma|\psi\rangle} \;,
    -- \]
    -- where \(|\psi\rangle\) is described by the RBM \(\mathcal{W}\) and
    -- \(|\sigma\rangle\in\{\downarrow,\uparrow\}^N\) is a spin configuration.
    -- We call this quantity /local energy/ \(E_{loc}(\sigma;\mathcal{W})\).
    -- \[
    -- E_{loc}(\sigma;\mathcal{W}) = \sum_i \frac{\langle\sigma|
    --    \hat{\boldsymbol\sigma}_i\cdot\hat{\boldsymbol\sigma}_{i+1}
    --    |\psi\rangle}{\langle\sigma|\psi\rangle}
    -- \]
    -- Now
    -- \[
    -- \begin{aligned}
    -- &\hat{\boldsymbol\sigma}_1\cdot\hat{\boldsymbol\sigma}_2
    --    |\uparrow\uparrow\rangle = |\uparrow\uparrow\rangle \;,\\
    -- &\hat{\boldsymbol\sigma}_1\cdot\hat{\boldsymbol\sigma}_2
    --    |\downarrow\downarrow\rangle = |\downarrow\downarrow\rangle \;,\\
    -- &\hat{\boldsymbol\sigma}_1\cdot\hat{\boldsymbol\sigma}_2
    --    |\uparrow\downarrow\rangle = |\uparrow\downarrow\rangle
    --      - 2|\downarrow\uparrow\rangle \;,\\
    -- &\hat{\boldsymbol\sigma}_1\cdot\hat{\boldsymbol\sigma}_2
    --    |\downarrow\uparrow\rangle = |\downarrow\uparrow\rangle
    --      - 2|\uparrow\downarrow\rangle \;.
    -- \end{aligned}
    -- \]
  -- , locEnergyHH1DOpen
  ) where

import Data.Complex(Complex(..))
import Numeric.LinearAlgebra(C, Matrix, Herm, (><), trustSym, kronecker)
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V

-- import PSO.Internal.Spin


one2D :: Matrix C
one2D = (2><2)
    [ 1.0,        0.0
    , 0.0,        1.0 ]

pauliX :: Matrix C
pauliX = (2><2)
    [ 0.0,        1.0
    , 1.0,        0.0 ]

pauliY :: Matrix C
pauliY = (2><2)
    [ 0.0,        0.0 :+ (-1.0)
    , 0.0 :+ 1.0, 0.0 ]

pauliZ :: Matrix C
pauliZ = (2><2)
    [ 1.0,        0.0
    , 0.0,        (-1.0) ]

rotate :: Int -> [a] -> [a]
rotate n xs = zipWith const (drop n (cycle xs)) xs

heisenberg1DOpen :: Int -> Herm C
heisenberg1DOpen n
  | n > 2 = trustSym $ heisenberg1DOpenImpl n pauliX
          + heisenberg1DOpenImpl n pauliY
          + heisenberg1DOpenImpl n pauliZ
  | n == 2 = trustSym $ pauliX `kronecker` pauliX
                      + pauliY `kronecker` pauliY
                      + pauliZ `kronecker` pauliZ
  | otherwise = undefined

heisenberg1DOpenImpl :: Int -> Matrix C -> Matrix C
heisenberg1DOpenImpl n pauli =
  let xs = (replicate (n-2) one2D) ++ (replicate 2 pauli)
      rotations = take (n-1) . iterate (rotate 1) $ xs
   in sum . map (foldr1 kronecker) $ rotations

-- locEnergyHH1DOpen ::
--      (Floating α, Eq α, Num (V.Vector α), LA.Numeric α)
--   => RBM α -> V.Vector α -> α
-- locEnergyHH1DOpen machine σ =
--   let θ = mkTheta machine σ
--       zipper i x y
--         | x == y    = 1
--         | otherwise = -1 + 2 * exp (logPoP2 machine θ σ (i, i + 1))
--    in V.sum $ V.izipWith zipper σ (V.tail σ)

