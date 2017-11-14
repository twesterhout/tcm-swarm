{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}

module PSO.Heisenberg
    ( one2D
    , pauliX
    , pauliY
    , pauliZ
    , heisenberg1DOpen
    ) where

import Data.Complex(Complex(..))
import Numeric.LinearAlgebra(C, Matrix, Herm, (><), trustSym, kronecker)
import qualified Numeric.LinearAlgebra as LA

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
  | otherwise = undefined

heisenberg1DOpenImpl :: Int -> Matrix C -> Matrix C
heisenberg1DOpenImpl n pauli =
  let xs = (replicate (n-2) one2D) ++ (replicate 2 pauli)
      rotations = take (n-1) . iterate (rotate 1) $ xs
   in sum . map (foldr1 kronecker) $ rotations




