{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_HADDOCK show-extensions #-}

module NQS.Internal.Types
  ( -- * Numeric types
    ℂ
  , ℝ

  -- , Estimate(..)
  -- , NormalError(..)
  -- , EnergyMeasurement(..)
  , RealOf
  -- , SRMeasurement(..)
  {-
  , withRbm
  , withRbmPure
  , withMRbm
  , withMRbmPure
  -}
    -- * Vectors and matrices
  , DenseVector
  , DenseMatrix
  , MDenseVector
  , MDenseMatrix
  , slice
  , withForeignPtrPrim
  , newVectorAligned
  , newDenseVector
  , newDenseMatrix

  , unsafeRow
  , unsafeColumn
  , unsafeWriteVector

  , Mutable(..)
  , Variant(..)
  , Orientation(..)
  , Transpose(..)
  , MatUpLo(..)
  , Sing(..)

  , CheckValid(..)
  , isValidVector
  , isValidMatrix
  , badVectorInfo
  , HasBuffer(..)
  , HasStride(..)
  , HasDim(..)

  , orientationOf
  , ToNative(..)

  , DenseWorkspace(..)
  , MCConfig(..)
  , defaultMCConfig
  , CGConfig(..)
  , SRConfig(..)

  , HasMc(..)
  , HasCg(..)
  , HasSteps(..)
  , HasRuns(..)
  , HasRate(..)
  , HasRestarts(..)
  , HasRegulariser(..)
  , HasMaxIter(..)
  , HasMagnetisation(..)

  , mapVectorM
  , mapMatrixM
  , zipWithVectorM
  , zipWithMatrixM
  -- , asTuple
  -- , fromTuple
  , FreezeThaw(..)

  , HasMean(..)
  , HasVar(..)
  -- , HasEnergy(..)
  -- , HasForce(..)
  -- , HasDerivatives(..)
  ) where

import Foreign.Storable
import Foreign.ForeignPtr

import Control.DeepSeq
import Control.Exception (assert)
import Control.Monad ((>=>), unless)
import Control.Monad.Primitive
import Control.Monad.ST.Strict
import Data.Complex
import Debug.Trace
import GHC.Generics (Generic)
import GHC.ForeignPtr (mallocPlainForeignPtrAlignedBytes)

import Data.Semigroup ((<>))
import Lens.Micro
import Lens.Micro.TH

import System.IO.Unsafe
import Data.Coerce
import Data.Bits

import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.ForeignPtr.Unsafe

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import Data.Vector.Storable (Vector, MVector(..))

import qualified Data.Vector.Generic as GV
import qualified Data.Vector as Boxed
import Data.Aeson
import qualified Data.Aeson.Types as Aeson

import Data.Singletons

import qualified Numerical.HBLAS.MatrixTypes   as HBLAS
import           Numerical.HBLAS.MatrixTypes    ( Variant(..)
                                                , Orientation(..)
                                                , Transpose(..)
                                                , MatUpLo(..)
                                                )
import           Numerical.HBLAS.BLAS.FFI       ( CBLAS_UPLOT(..) )

import Data.Kind
import Data.Proxy

deriving instance Generic Variant
deriving instance NFData Variant

deriving instance Generic Orientation
deriving instance NFData Orientation

class ToNative a b where
  encode :: a -> b

data family Mutable (v :: * -> *) :: * -> * -> *

-- | After some prototyping, I've concluded (possibly incorrectly) that a
-- single-precision 'Float' is enough for our purposes.
--
-- [This answer](https://stackoverflow.com/a/40538415) also mentions that 32-bit
-- floats should be enought for most neural network applications.
type ℝ = Float
type ℂ = Complex ℝ

data instance Sing (orient :: Orientation) where
  SRow :: Sing 'Row
  SColumn :: Sing 'Column

instance SingI 'Row where sing = SRow
instance SingI 'Column where sing = SColumn

instance SingKind Orientation where
  type Demote Orientation = Orientation
  fromSing SRow = Row
  fromSing SColumn = Column
  toSing x = case x of
    Row -> SomeSing SRow
    Column -> SomeSing SColumn

data DenseVector (variant :: Variant) (a :: *) = DenseVector
  { _denseVectorDim    :: {-# UNPACK #-}!Int
  , _denseVectorStride :: {-# UNPACK #-}!Int
  , _denseVectorBuffer :: {-# UNPACK #-}!(Vector a)
  } deriving (Show, Generic, NFData)

data DenseMatrix (orientation :: Orientation) (a :: *) = DenseMatrix
  { _denseMatrixRows   :: {-# UNPACK #-}!Int
  , _denseMatrixCols   :: {-# UNPACK #-}!Int
  , _denseMatrixStride :: {-# UNPACK #-}!Int
  , _denseMatrixBuffer :: {-# UNPACK #-}!(Vector a)
  } deriving (Show, Generic, NFData)

data instance Mutable (DenseVector variant) s a = MDenseVector
  { _mDenseVectorDim :: {-# UNPACK #-}!Int
  , _mDenseVectorStride :: {-# UNPACK #-}!Int
  , _mDenseVectorBuffer :: {-# UNPACK #-}!(MVector s a)
  } deriving (Generic, NFData)

type MDenseVector variant = Mutable (DenseVector variant)

data instance Mutable (DenseMatrix orientation) s a = MDenseMatrix
  { _mDenseMatrixRows :: {-# UNPACK #-}!Int
  , _mDenseMatrixCols :: {-# UNPACK #-}!Int
  , _mDenseMatrixStride :: {-# UNPACK #-}!Int
  , _mDenseMatrixBuffer :: {-# UNPACK #-}!(MVector s a)
  } deriving (Generic, NFData)

type MDenseMatrix orientation = Mutable (DenseMatrix orientation)

makeFields ''DenseVector
makeFields ''DenseMatrix


unsafeIndexVector :: Storable a => DenseVector 'Direct a -> Int -> a
{-# INLINE unsafeIndexVector #-}
unsafeIndexVector !(DenseVector _ stride buff) !i = V.unsafeIndex buff (i * stride)

unsafeIndexMatrix :: forall a orient. (Storable a, SingI orient)
                  => DenseMatrix orient a -> (Int, Int) -> a
{-# INLINE unsafeIndexMatrix #-}
unsafeIndexMatrix !(DenseMatrix _ _ stride buff) !(r, c) = V.unsafeIndex buff i
  where i = case (sing :: Sing orient) of
              SRow -> r * stride + c
              SColumn -> r + c * stride

unsafeReadVector :: (Storable a, PrimMonad m)
                 => MDenseVector 'Direct (PrimState m) a -> Int -> m a
{-# INLINE unsafeReadVector #-}
unsafeReadVector !(MDenseVector _ stride buff) !i = MV.unsafeRead buff (i * stride)

unsafeReadMatrix :: forall a m orient. (Storable a, PrimMonad m, SingI orient)
                 => MDenseMatrix orient (PrimState m) a -> (Int, Int) -> m a
{-# INLINE unsafeReadMatrix #-}
unsafeReadMatrix !(MDenseMatrix _ _ stride buff) !(r, c) = MV.unsafeRead buff i
  where i = case (sing :: Sing orient) of
              SRow -> r * stride + c
              SColumn -> r + c * stride

unsafeWriteVector :: (Storable a, PrimMonad m)
                  => MDenseVector 'Direct (PrimState m) a -> Int -> a -> m ()
{-# INLINE unsafeWriteVector #-}
unsafeWriteVector !(MDenseVector _ stride buff) !i !x = MV.unsafeWrite buff (i * stride) x

unsafeWriteMatrix :: forall a m orient. (Storable a, PrimMonad m, SingI orient)
                  => MDenseMatrix orient (PrimState m) a -> (Int, Int) -> a -> m ()
{-# INLINE unsafeWriteMatrix #-}
unsafeWriteMatrix !(MDenseMatrix _ _ stride buff) !(r, c) !x = MV.unsafeWrite buff i x
  where i = case (sing :: Sing orient) of
              SRow -> r * stride + c
              SColumn -> r + c * stride

-- | Returns a row of a matrix.
unsafeRow
  :: forall orient s a
   . (Storable a, SingI orient)
  => Int
  -> MDenseMatrix orient s a
  -> MDenseVector 'Direct s a
{-# INLINE unsafeRow #-}
unsafeRow !i !(MDenseMatrix rows cols stride buff) = assert False $
  assert (i < rows) $ case (sing :: Sing orient) of
    SRow    -> MDenseVector cols 1 (MV.slice (i * stride) cols buff)
    SColumn -> MDenseVector cols stride (MV.slice i (cols * stride) buff)




-- | Returns a column of a matrix.
unsafeColumn
  :: forall orient s a
   . (Storable a, SingI orient)
  => Int
  -> MDenseMatrix orient s a
  -> MDenseVector 'Direct s a
{-# INLINE unsafeColumn #-}
unsafeColumn !i !(MDenseMatrix rows cols stride buff) =
  assert (i < cols) $ case (sing :: Sing orient) of
    SRow    -> if rows == 1
                 then MDenseVector 1 1 (MV.slice i 1 buff)
                 else MDenseVector rows stride (MV.slice i ((rows - 1)* stride + 1) buff)
    SColumn -> MDenseVector rows 1 (MV.slice (i * stride) rows buff)




data DenseWorkspace s a = DenseWorkspace
  { _denseWorkspaceForce       :: !(MDenseVector 'Direct s a) -- ^ Force
  , _denseWorkspaceDerivatives :: !(MDenseMatrix 'Row s a)    -- ^ Derivatives
  , _denseWorkspaceDelta       :: !(MDenseVector 'Direct s a) -- ^ Old delta
  } deriving (Generic, NFData)


-- | Configuration for Monte-Carlo sampling.
data MCConfig =
  MCConfig
    { _mCConfigSteps :: {-# UNPACK #-}!(Int, Int, Int)
        -- ^ A range of steps for a single run. It is very similar to Python's
        -- [range](). @(low, high, step)@ corresponds to [low, low + step, low +
        -- 2 * step, ..., high).
        --
        -- /Note:/ only positive @step@s are supported.
    , _mCConfigThreads :: {-# UNPACK #-}!(Int, Int, Int)
    , _mCConfigRuns :: {-# UNPACK #-}!Int
        -- ^ Number of Monte-Carlo runs to perform.
        --
        -- /Note:/ for optimal work scheduling, make that the total number of
        -- threads is divisible by the number of runs.
    , _mCConfigFlips :: {-# UNPACK #-}!Int
        -- ^ Number of spin-flips to do at each step.
    , _mCConfigRestarts :: {-# UNPACK #-}!Int
        -- ^ Allowed number of restarts.
        --
        -- A restart happens when the function computing local energy notices
        -- that a particular spin-flip results in a spin configuration with
        -- significantly higher probability. Monte-Carlo sampler is then reset
        -- and this new spin configuration is used as the initial one.
    , _mCConfigMagnetisation :: !(Maybe Int)
        -- ^ Specifies the magnetisation over which to sample.
    }

makeFields ''MCConfig

defaultMCConfig :: MCConfig
defaultMCConfig = MCConfig (1000, 11000, 1) (4, 1, 1) 4 2 5 Nothing

-- | Configuration for the Conjugate Gradient solver.
data CGConfig a =
  CGConfig { _cGConfigMaxIter :: {-# UNPACK #-}!Int
               -- ^ Maximum number of iterations.
           , _cGConfigTol :: !a
               -- ^ Tolerance.
           }

makeFields ''CGConfig

-- | Configuration for the Stochastic Reconfiguration algorithm.
data SRConfig a =
  SRConfig
    { _sRConfigMaxIter :: !Int
        -- ^ Maximum number of iterations to perform.
    , _sRConfigRegulariser :: !(Maybe (Int -> ℂ))
        -- ^ Regulariser for the S matrix. If it is @'Just' f@,
        -- then @λ = f i@ where @i@ is the currect iteration is used to
        -- regularise the matrix S according to @S <- S + λ1@.
    , _sRConfigRate :: !(Int -> ℂ)
        -- ^ Learning rate as a function of the iteration.
    , _sRConfigCg :: !(CGConfig ℝ)
        -- ^ Configuration for the Monte-Carlo sampling.
    , _sRConfigMc :: !MCConfig
        -- ^ Configuration for the Conjugate Gradient solver.
    }

-- | Returns a slice of the vector.
slice
  :: forall a s
   . Storable a
  => Int -- ^ Start index
  -> Int -- ^ Length
  -> MDenseVector 'Direct s a -- ^ Source vector
  -> MDenseVector 'Direct s a -- ^ Slice
slice !i !n !(MDenseVector size stride buff) =
  MDenseVector n stride (MV.slice (stride * i) (stride * n) buff)

makeFields ''SRConfig

type family RealOf a :: *

type instance RealOf Float = Float
type instance RealOf Double = Double
type instance RealOf (Complex a) = a

mallocVectorAligned :: forall a. Storable a => Int -> Int -> IO (ForeignPtr a)
mallocVectorAligned n alignment =
  mallocPlainForeignPtrAlignedBytes (n * sizeOf (undefined :: a)) alignment

newVectorAligned
  :: (Storable a, PrimMonad m) => Int -> Int -> m (MVector (PrimState m) a)
newVectorAligned n alignment =
  unsafePrimToPrim $! MVector n <$> mallocVectorAligned n alignment

newDenseVector
  :: (Storable a, PrimMonad m) => Int -> m (MDenseVector 'Direct (PrimState m) a)
newDenseVector n = MDenseVector n 1 <$> newVectorAligned n 64

-- | Default alignment used when allocating new vectors and matrices.
defaultAlignment :: Int
defaultAlignment = 64

roundUpTo :: Int -> Int -> Int
{-# INLINE roundUpTo #-}
roundUpTo !alignment !n = assert (isValidAlignment alignment && n >= 0) $
  (n + alignment - 1) .&. complement (alignment - 1)
  where isValidAlignment !x = x > 0 && (x .&. (x - 1) == 0)

newDenseMatrix
  :: forall orient a m. (Storable a, SingI orient, PrimMonad m)
  => Int
  -> Int
  -> m (MDenseMatrix orient (PrimState m) a)
newDenseMatrix rows cols = case (sing :: Sing orient) of
  SRow ->
    let ldim = assert (defaultAlignment `mod` sizeOf (undefined :: a) == 0) $
                roundUpTo (defaultAlignment `div` sizeOf (undefined :: a)) cols
    in  MDenseMatrix rows cols ldim
          <$> newVectorAligned (rows * ldim) defaultAlignment
  SColumn ->
    let ldim = roundUpTo defaultAlignment rows
    in  MDenseMatrix rows cols ldim
          <$> newVectorAligned (ldim * cols) defaultAlignment

class HasOrientation s a | s -> a where
  orientationOf :: s -> a

instance SingI orient
  => HasOrientation (DenseMatrix orient a) Orientation where
    orientationOf _ = fromSing (sing :: Sing orient)

instance SingI orient
  => HasOrientation (Mutable (DenseMatrix orient) s a) Orientation where
    orientationOf _ = fromSing (sing :: Sing orient)



isValidMatrix :: Orientation -- ^ Row- vs. column-major layout
              -> Int -- ^ Number of rows
              -> Int -- ^ Number of columns
              -> Int -- ^ Stride
              -> Int -- ^ Buffer size
              -> Bool
isValidMatrix orient rows cols i size =
  i >= 0 && (rows == 0 && cols == 0 || rows > 0 && cols > 0 && validAccesses orient)
  where validAccesses Row    = (rows - 1) * i < size && cols <= i
        validAccesses Column = (cols - 1) * i < size && rows <= i

-- | Constructs a nice message describing the problem in the BLAS vector.
badMatrixInfo :: String -- ^ Function name
              -> String -- ^ Argument name
              -> Int -- ^ Number of rows
              -> Int -- ^ Number of columns
              -> Int -- ^ Stride
              -> Int -- ^ Size of the underlying buffer
              -> String -- ^ Error message
badMatrixInfo funcName argName rows cols i size
  | rows < 0 = preamble <> " has a negative number of rows: " <> show rows <> "."
  | cols < 0 = preamble <> " has a negative number of columns: " <> show cols <> "."
  | i < 0 = preamble <> " has a negative stride: " <> show i <> "."
  | otherwise = preamble <> " has invalid range of accesses: #rows = " <> show rows <>
    ", #cols = " <> show cols <> ", stride = " <> show i <> ", bufferSize = " <> show size <> "."
  where preamble = funcName <> ": " <> argName

-- | Returns whether strides and dimensions are consistent.
isValidVector :: Int -- ^ Logical dimension
              -> Int -- ^ Stride
              -> Int -- ^ Buffer size
              -> Bool
isValidVector n i size = i >= 0 && (n == 0 || n > 0 && (n - 1) * i < size)

-- | Constructs a nice message describing the problem in the BLAS vector.
badVectorInfo :: String -- ^ Function name
              -> String -- ^ Argument name
              -> Int -- ^ Logical vector dimension
              -> Int -- ^ Vector stride
              -> Int -- ^ Size of the underlying buffer
              -> String -- ^ Error message
badVectorInfo funcName argName n i size
  | n < 0 = preamble <> " has a negative logical dimension: " <> show n <> "."
  | i < 0 = preamble <> " has a negative stride: " <> show i <> "."
  | otherwise = preamble <> " has invalid range of accesses: dim = " <> show n <>
                ", stride = " <> show i <> ", bufferSize = " <> show size <> "."
  where preamble = funcName <> ": " <> argName

class CheckValid a where
  assertValid :: String -> String -> a -> b -> b

instance Storable a => CheckValid (DenseVector variant a) where
  assertValid funcName argName !(DenseVector dim stride (V.length -> size))
    | isValidVector dim stride size = id
    | otherwise = error $! badVectorInfo funcName argName dim stride size

instance Storable a => CheckValid (Mutable (DenseVector variant) s a) where
  assertValid funcName argName !(MDenseVector dim stride (MV.length -> size))
    | isValidVector dim stride size = id
    | otherwise = error $! badVectorInfo funcName argName dim stride size

instance (Storable a, SingI orient) => CheckValid (DenseMatrix orient a) where
  assertValid funcName argName !m@(DenseMatrix rows cols stride (V.length -> size))
    | isValidMatrix (orientationOf m) rows cols stride size = id
    | otherwise = error $! badMatrixInfo funcName argName rows cols stride size

instance (Storable a, SingI orient) => CheckValid (Mutable (DenseMatrix orient) s a) where
  assertValid funcName argName !m@(MDenseMatrix rows cols stride (MV.length -> size))
    | isValidMatrix (orientationOf m) rows cols stride size = id
    | otherwise = error $! badMatrixInfo funcName argName rows cols stride size

instance Storable a => HasBuffer (MDenseVector variant s a) (MVector s a) where
  buffer inj (MDenseVector dim stride buf) = MDenseVector dim stride <$> inj buf

instance Storable a => HasBuffer (MDenseMatrix orientation s a) (MVector s a) where
  buffer inj (MDenseMatrix rows cols stride buf) = MDenseMatrix rows cols stride <$> inj buf

instance Storable a => HasStride (MDenseVector variant s a) Int where
  stride inj (MDenseVector dim stride buf) = (\x -> MDenseVector dim x buf) <$> inj stride

instance Storable a => HasStride (MDenseMatrix orientation s a) Int where
  stride inj (MDenseMatrix rows cols stride buf) = (\x -> MDenseMatrix rows cols x buf) <$> inj stride

instance Storable a => HasDim (MDenseVector variant s a) Int where
  dim inj (MDenseVector dim stride buf) = (\x -> MDenseVector x stride buf) <$> inj dim

instance Storable a => HasDim (MDenseMatrix orientation s a) (Int, Int) where
  dim inj (MDenseMatrix rows cols stride buf) =
    (\(x, y) -> MDenseMatrix x y stride buf) <$> inj (rows, cols)

instance Storable a => HasDim (DenseMatrix orientation a) (Int, Int) where
  dim inj (DenseMatrix rows cols stride buf) =
    (\(x, y) -> DenseMatrix x y stride buf) <$> inj (rows, cols)

class FreezeThaw (v :: * -> *) a where
  unsafeFreeze :: PrimMonad m => (Mutable v) (PrimState m) a -> m (v a)
  unsafeThaw :: PrimMonad m => v a -> m ((Mutable v) (PrimState m) a)

instance Storable a => FreezeThaw (DenseVector variant) a where
  unsafeFreeze (MDenseVector dim stride mv) = DenseVector dim stride <$> V.unsafeFreeze mv
  unsafeThaw (DenseVector dim stride mv) = MDenseVector dim stride <$> V.unsafeThaw mv

instance Storable a => FreezeThaw (DenseMatrix orientation) a where
  unsafeFreeze (MDenseMatrix rows cols stride mv) = DenseMatrix rows cols stride <$> V.unsafeFreeze mv
  unsafeThaw (DenseMatrix rows cols stride v) = MDenseMatrix rows cols stride <$> V.unsafeThaw v


touchForeignPtrPrim :: PrimMonad m => ForeignPtr a -> m ()
{-# NOINLINE touchForeignPtrPrim #-}
touchForeignPtrPrim fp = unsafeIOToPrim $! touchForeignPtr fp

withForeignPtrPrim :: PrimMonad m => ForeignPtr a -> (Ptr a -> m b) -> m b
{-# INLINE withForeignPtrPrim #-}
withForeignPtrPrim p func = do r <- func (unsafeForeignPtrToPtr p)
                               touchForeignPtrPrim p
                               return r

mapVectorM ::
     (PrimMonad m, Storable a, Storable b)
  => (a -> m b)
  -> MDenseVector 'Direct (PrimState m) a
  -> MDenseVector 'Direct (PrimState m) b
  -> m ()
mapVectorM f x y =
  assertValid "NQS.Rbm.zipWithVectorM" "x" x $
  assertValid "NQS.Rbm.zipWithVectorM" "x" y $
  assert (x ^. dim == y ^. dim) $ go 0
  where
    n = x ^. dim
    go !i
      | i == n = return ()
      | otherwise = do
        xi <- unsafeReadVector x i
        yi <- f xi
        unsafeWriteVector y i yi
        go (i + 1)

zipWithVectorM ::
     (PrimMonad m, Storable a, Storable b, Storable c)
  => (a -> b -> m c)
  -> MDenseVector 'Direct (PrimState m) a
  -> MDenseVector 'Direct (PrimState m) b
  -> MDenseVector 'Direct (PrimState m) c
  -> m ()
zipWithVectorM f x y z =
  assertValid "NQS.Rbm.zipWithVectorM" "x" x $
  assertValid "NQS.Rbm.zipWithVectorM" "x" y $
  assertValid "NQS.Rbm.zipWithVectorM" "x" z $
  assert (y ^. dim == n && z ^. dim == n) $ go 0
  where
    n = x ^. dim
    go !i
      | i == n = return ()
      | otherwise = do
        xi <- unsafeReadVector x i
        yi <- unsafeReadVector y i
        zi <- f xi yi
        unsafeWriteVector z i zi
        go (i + 1)

mapMatrixM ::
     forall m a b orientX orientY.
     (PrimMonad m, Storable a, Storable b, SingI orientX, SingI orientY)
  => (a -> m b)
  -> MDenseMatrix orientX (PrimState m) a
  -> MDenseMatrix orientY (PrimState m) b
  -> m ()
mapMatrixM f x y =
  assertValid "NQS.Rbm.zipWithMatrixM" "x" x $
  assertValid "NQS.Rbm.zipWithMatrixM" "y" y $
  assert (x ^. dim == y ^. dim) $
  case (sing :: Sing orientY) of
    SRow -> stepperRow 0 0
    SColumn -> stepperColumn 0 0
  where
    !(n, m) = x ^. dim
    stepperRow !i !j
      | j < m && i < n = go i j >> stepperRow i (j + 1)
      | i < n = stepperRow (i + 1) 0
      | otherwise = return ()
    stepperColumn !i !j
      | i < n && j < m = go i j >> stepperColumn (i + 1) j
      | j < m = stepperColumn 0 (j + 1)
      | otherwise = return ()
    go !i !j = do
      xij <- unsafeReadMatrix x (i, j)
      yij <- f xij
      unsafeWriteMatrix y (i, j) yij

zipWithMatrixM :: forall m a b c orientX orientY orientZ.
     (PrimMonad m, Storable a, Storable b, Storable c, SingI orientX, SingI orientY, SingI orientZ)
  => (a -> b -> m c)
  -> MDenseMatrix orientX (PrimState m) a
  -> MDenseMatrix orientY (PrimState m) b
  -> MDenseMatrix orientZ (PrimState m) c
  -> m ()
zipWithMatrixM f x y z =
  assertValid "NQS.Rbm.zipWithMatrixM" "x" x $
  assertValid "NQS.Rbm.zipWithMatrixM" "y" y $
  assertValid "NQS.Rbm.zipWithMatrixM" "z" z $
  assert (y ^. dim == (n, m) && z ^. dim == (n, m)) $
    case (sing :: Sing orientZ) of
      SRow -> stepperRow 0 0
      SColumn -> stepperColumn 0 0
  where
    !(n, m) = x ^. dim
    stepperRow !i !j
      | j < m && i < n = go i j >> stepperRow i (j + 1)
      | i < n = stepperRow (i + 1) 0
      | otherwise = return ()
    stepperColumn !i !j
      | i < n && j < m = go i j >> stepperColumn (i + 1) j
      | j < m = stepperColumn 0 (j + 1)
      | otherwise = return ()
    go !i !j = do
      xij <- unsafeReadMatrix x (i, j)
      yij <- unsafeReadMatrix y (i, j)
      zij <- f xij yij
      unsafeWriteMatrix z (i, j) zij

{-
withRbm :: PrimMonad m => Rbm a -> (Ptr (RbmCore a) -> m b) -> m b
withRbm (Rbm p) func = withForeignPtrPrim p func

withRbmPure :: Rbm a -> (Ptr (RbmCore a) -> b) -> b
withRbmPure (Rbm p) func = unsafePerformIO $! withForeignPtrPrim p (return . func)

withMRbm :: PrimMonad m => MRbm (PrimState m) a -> (Ptr (RbmCore a) -> m b) -> m b
withMRbm (MRbm p) func = withForeignPtrPrim p func

withMRbmPure :: MRbm s a -> (Ptr (RbmCore a) -> b) -> b
withMRbmPure (MRbm p) func = unsafePerformIO $! withForeignPtr p (return . func)
-}


data Estimate e a = Estimate
  { estPoint :: !a
  , estError :: !(e a)
  } deriving (Generic, NFData)

newtype NormalError a = NormalError a
  deriving (Eq, Ord, Read, Show, Generic, NFData)

data EnergyMeasurement a = EnergyMeasurement
  { _energyMeasurementMean :: !a
  , _energyMeasurementVar :: !a
  } deriving (Generic)

makeFields ''EnergyMeasurement

-- | Simple for loop. Counts from /start/ to /end/-1.
for :: Monad m => Int -> Int -> (Int -> m ()) -> m ()
for !start !end f = assert (start <= end) $ loop start
 where
  loop !i | i == end  = return ()
          | otherwise = f i >> loop (i + 1)
{-# INLINE for #-}


for' :: Monad m => a -> Int -> (a -> Int -> m a) -> m a
for' !x0 !n f = assert (n >= 0) $ loop x0 0
 where
  loop !x !i | i == n    = return x
             | otherwise = f x i >>= \x' -> loop x' (i + 1)
{-# INLINE for' #-}



generateVectorM
  :: forall a m
   . (Monad m, Storable a)
  => Int
  -> (Int -> m a)
  -> m (DenseVector 'Direct a)
{-# NOINLINE generateVectorM #-}
generateVectorM !n f
  | n < 0     = error $! "generateVectorM: Invalid dimension: " <> show n
  | n == 0    = return $! DenseVector 0 0 V.empty
  | otherwise = go (runST $ newDenseVector n >>= unsafeFreeze)
 where
  go :: DenseVector 'Direct a -> m (DenseVector 'Direct a)
  go !v' = for' v' n $ \v i -> f i >>= \x -> return (set v i x)
  set !v !i !x = runST $ do
    !v' <- unsafeThaw v
    unsafeWriteVector v' i x
    unsafeFreeze v'

generateMatrixM
  :: forall orient m a. (Monad m, Storable a, SingI orient)
  => Int
  -> Int
  -> (Int -> Int -> m a)
  -> m (DenseMatrix orient a)
{-# NOINLINE generateMatrixM #-}
generateMatrixM !rows !cols f
  | rows < 0 || cols < 0 = error $! "generateMatrixM: Invalid dimensions: " <> show (rows, cols)
  | rows == 0 || cols == 0 = return $! DenseMatrix 0 0 0 V.empty
  | otherwise = go (runST $ newDenseMatrix rows cols >>= unsafeFreeze)
 where
  go :: DenseMatrix orient a -> m (DenseMatrix orient a)
  go !m'' = case (sing :: Sing orient) of
    SRow -> for' m'' rows $ \m' i -> for' m' cols $ \m j ->
              f i j >>= \x -> return (set m i j x)
    SColumn -> for' m'' cols $ \m' j -> for' m' rows $ \m i ->
                 f i j >>= \x -> return (set m i j x)
  set !m !i !j !x = runST $ do
    !m' <- unsafeThaw m
    unsafeWriteMatrix m' (i, j) x
    unsafeFreeze m'

instance (Storable a, FromJSON a) => FromJSON (DenseVector 'Direct a) where
  parseJSON = withArray "vector elements (i.e. Array)" $ \v -> do
    generateVectorM (GV.length v) $ \i -> parseJSON (v `GV.unsafeIndex` i)

instance ToJSON a => ToJSON (Complex a) where
  toJSON (x :+ y) = toJSON [x, y]
  toEncoding (x :+ y) = toEncoding [x, y]

instance FromJSON a => FromJSON (Complex a) where
  parseJSON = withArray "Complex" $ \v ->
    (:+) <$> parseJSON (v GV.! 0)
         <*> parseJSON (v GV.! 1)

instance (Storable a, FromJSON a, SingI orient) => FromJSON (DenseMatrix orient a) where
  parseJSON = withArray "Matrix rows" $ \matrix -> do
    (n, m) <- checkDimensions matrix
    generateMatrixM n m $ \i j ->
      flip (withArray "a matrix row") (matrix `GV.unsafeIndex` i) $ \row ->
        parseJSON (row `GV.unsafeIndex` j)
    where
      checkDimensions x@(GV.length -> n)
        | n == 0 = return (0, 0)
        | otherwise = do
          !m <- withArray "first row (i.e. Array)" (return . GV.length) (GV.unsafeHead x)
          for 0 n $ \i ->
            flip (withArray "a matrix row (i.e. Array)") (x `GV.unsafeIndex` i) $ \row ->
              unless (GV.length row == m) $ fail $! "Matrix row #" <> show i <> " has wrong dimension."
          return (n, m)

vectorAsVector :: Storable a => DenseVector 'Direct a -> Boxed.Vector a
vectorAsVector !(DenseVector n stride buff) =
  Boxed.generate n (\i -> buff `V.unsafeIndex` (i * stride))

-- | Returns a row of a matrix.
unsafeRowAsVector
  :: forall orient a
   . (Storable a, SingI orient)
  => Int
  -> DenseMatrix orient a
  -> Boxed.Vector a
unsafeRowAsVector !i !(DenseMatrix rows cols stride buff) =
  assert (i < rows) $ case (sing :: Sing orient) of
    SRow    -> GV.generate cols (\j -> buff `V.unsafeIndex` (i * stride + j))
    SColumn -> GV.generate cols (\j -> buff `V.unsafeIndex` (i + j * stride))

instance (Storable a, ToJSON a) => ToJSON (DenseVector 'Direct a) where
  toEncoding = toEncoding . vectorAsVector

instance (Storable a, ToJSON a, SingI orient) => ToJSON (DenseMatrix orient a) where
  toEncoding x = toEncoding $ Boxed.generate (x ^. dim . _1) (\i -> unsafeRowAsVector i x)



