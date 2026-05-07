#!/usr/bin/bash
python save_plot_data.py 1 4 0
python save_plot_data.py 1 4 1
for i in 3 6 15
do
  for j in 4
  do
    python save_plot_data.py $i $j
  done
done