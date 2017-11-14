#!/bin/bash

PROJECT_ROOT="."


pushd "$PROJECT_ROOT"

cabal configure \
	--package-db=clear \
	--package-db=global \
	--package-db=$(stack path --snapshot-pkg-db) \
	--package-db=$(stack path --local-pkg-db)

cabal haddock \
	--html-location='http://hackage.haskell.org/packages/archive/$pkg/latest/doc/html'

cp -r "dist/doc/"* "doc/"
rm -r "dist"
popd
