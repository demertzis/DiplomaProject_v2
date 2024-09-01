#!/usr/bin/bash
python save_plot_data.py 1 3 0
python save_plot_data.py 1 3 1
for i in 3 6 15
do
  python save_plot_data.py $i 4
  python save_plot_data.py $i 5
  python save_plot_data.py $i 6
  python save_plot_data.py $i 7
done