#!/usr/bin/gnuplot --persist

Stem(s)       = system(sprintf("echo '%s' | sed -E 's/\\.[^.]*$//'", s))

Move(a, b)    = system(sprintf("mv -v '%s' '%s'", a, b))

Compile(s)    = system(sprintf("pdflatex '%s' 1>&2 && rm -f '%s' '%s' '%s' 1>&2 " \
    . "&& echo '%s'", s, Stem(s) . '.log', Stem(s) . '.aux', \
    Stem(s) . '-inc-eps-converted-to.pdf', Stem(s) . '.pdf'))

Crop(s)       = system(sprintf("pdfcrop '%s' 1>&2 && echo '%s'", s, \
    Stem(s) . '-crop.pdf'))

PdfToPng(pdffile, pngfile) = system(sprintf("gs -sDEVICE=png16m " \
    . "-dTextAlphaBits=4 -dNumRenderingThreads=4 -r600 -o '%s' '%s'", \
    pngfile, pdffile))

bin(x, width) = width * floor(x / width + .5)
max(x, y) = (x >= y) ? x : y
min(x, y) = (x >= y) ? y : x
round(x, d) = floor(10.**d * x + 0.5) / 10.**d
