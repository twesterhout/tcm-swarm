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
  , fromPyFile
  ) where

import Data.Complex
import Data.Maybe
import Control.Monad
import Control.Applicative
import Data.List (transpose)
import Data.Text (Text)
import Data.Text.IO (hGetLine)
import qualified Data.Text as T
import Data.Attoparsec.Text
import qualified Data.Vector.Storable as V
import GHC.Float
import System.IO (withFile, Handle, IOMode(..), FilePath)
import PSO.Neural

fromPyComplex :: Parser (Complex Double)
fromPyComplex = do
  x <- double
  skipMany space
  sign <- satisfy (\c -> c == '+' || c == '-')
  skipMany space
  y <- double
  char 'j'
  if sign == '+'
     then return (x :+ y)
     else return (x :+ (-y))

between :: Parser start -> Parser stop -> Parser a -> Parser a
between start stop p = do
  start
  x <- p
  stop
  return x

fromPyList :: Parser a -> Parser [a]
fromPyList p =
  let stripSpaces = between (skipMany space) (skipMany space)
      element = stripSpaces $ (between (char '(') (char ')') (stripSpaces p))
                              <|> (stripSpaces p)
   in between (char '[') (char ']') (element `sepBy'` char ',')

readVector :: Text -> Either String (V.Vector (Complex Float))
readVector txt =
  V.fromList <$> fmap (\(x :+ y) -> double2Float x :+ double2Float y)
             <$> parseOnly (fromPyList fromPyComplex <* endOfInput) txt

readMatrix :: Text -> Either String (V.Vector (Complex Float))
readMatrix txt =
  V.fromList <$> fmap (\(x :+ y) -> double2Float x :+ double2Float y)
             <$> concat
             <$> transpose
             <$> parseOnly (fromPyList (fromPyList fromPyComplex) <* endOfInput) txt

fromPyFile :: FilePath -> IO (Rbm (Complex Float))
fromPyFile name = withFile name ReadMode toRbm
  where toRight (Right x) = return x
        toRight (Left x)  = error x
        toRbm h = do
          a <- toRight <$> readVector <$> hGetLine h
          b <- toRight <$> readVector <$> hGetLine h
          w <- toRight <$> readMatrix <$> hGetLine h
          join $ mkRbm <$> a <*> b <*> w

