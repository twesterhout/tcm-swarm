#!/usr/bin/gnuplot -p


datafile = "Energies.dat"


set terminal pngcairo \
    enhanced color font "DejaVu Sans Mono, 10" size 800,600
set output "Energies.png"

set ylabel "〈H〉"
set xlabel "Iteration"

plot for [i = 1:*] \
    "Energies.dat" using 0:i \
    with points pointtype 7 pointsize 0.3 \
    linecolor rgb "#303030" notitle

set output
set terminal x11
