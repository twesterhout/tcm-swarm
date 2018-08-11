{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ViewPatterns #-}

-- |
-- Module      : PSO.FromPython
-- Description : Communication with Python
-- Copyright   : (c) Tom Westerhout, 2018
-- License     : BSD3
-- Maintainer  : t.westerhout@student.ru.nl
-- Stability   : experimental
module PSO.FromPython
  ( fromPyComplex
  , fromPyList
  , readVector
  , readMatrix
  ) where

import Control.Applicative
import Control.Monad
import Debug.Trace
import Data.Attoparsec.Text
import Data.Complex
import Data.List (transpose)
import Data.Maybe
import Data.Text (Text)
import qualified Data.Text as T
import Data.Text.IO (hGetLine)
import qualified Data.Vector.Storable as V
import GHC.Float
import System.IO (FilePath, Handle, IOMode(..), withFile)

between :: Parser start -> Parser stop -> Parser a -> Parser a
between start stop p = do
  start
  x <- p
  stop
  return x

fromPyComplex :: Parser (Complex Double)
fromPyComplex = pBoth <|> pImag
  where
    pImag = do
      y <- double
      char 'j'
      return (0 :+ y)
    pBoth =
      between (char '(') (char ')') $ do
        x <- double
        y <- double
        char 'j'
        return (x :+ y)

fromPyList :: Parser a -> Parser [a]
fromPyList p = between (char '[') (char ']') (p `sepBy'` string ", ")

readVector :: Text -> Either String (V.Vector (Complex Float))
readVector txt =
  V.fromList <$> fmap (\(x :+ y) -> double2Float x :+ double2Float y) <$>
  parseOnly (fromPyList fromPyComplex <* endOfInput) txt

readMatrix :: Text -> Either String (V.Vector (Complex Float))
readMatrix txt =
  V.fromList <$> fmap (\(x :+ y) -> double2Float x :+ double2Float y) <$> concat <$>
  transpose <$>
    parseOnly (fromPyList (fromPyList fromPyComplex >>= \xs -> trace (show xs) (return xs)) ) txt
