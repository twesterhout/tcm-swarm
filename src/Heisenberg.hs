{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}

module Heisenberg
    ( one2D
    , pauliX
    , pauliY
    , pauliZ
    , heisenberg1D3Open
    , heisenberg1D4Open
    ) where

import Numeric.LinearAlgebra (Complex(..), R, C, Matrix, (><), kronecker, scale)
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

heisenberg1D3Open :: Matrix C
heisenberg1D3Open =
    let sum = (pauliX `kronecker` pauliX `kronecker` one2D)
            + (pauliY `kronecker` pauliY `kronecker` one2D)
            + (pauliZ `kronecker` pauliZ `kronecker` one2D)

            + (one2D  `kronecker` pauliX `kronecker` pauliX)
            + (one2D  `kronecker` pauliY `kronecker` pauliY)
            + (one2D  `kronecker` pauliZ `kronecker` pauliZ)

            + (one2D  `kronecker` one2D  `kronecker` pauliX)
            + (one2D  `kronecker` one2D  `kronecker` pauliY)
            + (one2D  `kronecker` one2D  `kronecker` pauliZ)
    in sum

heisenberg1D4Open :: Matrix C
heisenberg1D4Open =
    let sum = (pauliX `kronecker` pauliX `kronecker` one2D  `kronecker` one2D)
            + (pauliY `kronecker` pauliY `kronecker` one2D  `kronecker` one2D)
            + (pauliZ `kronecker` pauliZ `kronecker` one2D  `kronecker` one2D)

            + (one2D  `kronecker` pauliX `kronecker` pauliX `kronecker` one2D)
            + (one2D  `kronecker` pauliY `kronecker` pauliY `kronecker` one2D)
            + (one2D  `kronecker` pauliZ `kronecker` pauliZ `kronecker` one2D)

            + (one2D  `kronecker` one2D  `kronecker` pauliX `kronecker` pauliX)
            + (one2D  `kronecker` one2D  `kronecker` pauliY `kronecker` pauliY)
            + (one2D  `kronecker` one2D  `kronecker` pauliZ `kronecker` pauliZ)
    in sum

