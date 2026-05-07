import itertools
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# scratch = True
scratch = False
pretrained = True
# pretrained = False

show_running_mean = True
# show_running_mean = False



# func = "scratch" if scratch else "pretrained"
func_list = []
if scratch:
    func_list.append("scratch")
if pretrained:
    func_list.append("pretrained")

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 18,
        }

fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
ax.set_xlabel('Epochs', fontdict=font)
ax.set_ylabel('Total Reward for All Stations ({})'.format(u'\u20AC'), fontdict=font)

network_initialization = "" if pretrained and scratch \
                            else "Starting from Random Initialized Networks" if scratch \
                            else "Starting from Pretrained Networks" if pretrained else ""
ax.set_title(('Progression of Total Average Daily Cost' +
             network_initialization).title(), fontdict=font)
ax.set_ylim(1000, 0)


single_fmt = ["-", ":"] if len(func_list) == 2 else ["-"]
fmt_list = [i[0] + i[1] for i in itertools.product(['r', 'b', 'g', 'm', 'y'], single_fmt)]
main_dir = 'train_eval_records_2/15_AGENTS'
func_dirs = os.listdir(main_dir)
for dir in func_dirs:
    for func in func_list:
        with open('/'.join([main_dir, dir,  func + '720eval_record.json']), 'r') as file:
            data = json.loads(file.read())
            # with open('/'.join(["train_eval_records", "3_AGENTS", dir, func + '200eval_record.json']), 'r') as file:
            #     data += (json.loads(file.read()))
            new_data = list(map(lambda x: -10.0 * x, data))
            ax.set_ylim(min(ax.get_ylim()[0], min(new_data)-100), max(ax.get_ylim()[1], max(new_data)+100))
            fmt=''
            line = ax.plot(np.arange(0, len(new_data)),
                           new_data,
                           fmt_list.pop(0),
                           label='-'.join([dir.replace('_', ' '), func]).title())


for line in fig.gca().get_lines():
    data = line.get_ydata()
    min_x, min_y = np.argmin(data), np.min(data)
    # ax.scatter(min_x, min_y, c='c')
    ax.annotate(round(min_y, 3), (min_x, min_y), xytext=(min_x+4, min_y-150), fontsize='large',  arrowprops=dict(facecolor='black', shrink=0.0, headwidth= 6,width=2))
    # plt.text(min_x, min_y, "{}".format(min_y), fontsize=12)

if show_running_mean:
    for line in fig.gca().get_lines():
        line.set_alpha(0.3)
        mean = 50
        running_mean = np.convolve( np.pad(line.get_ydata(), (mean-1, 0),mode='edge'), np.ones(mean) / mean, mode='valid')
        # new_line = ax.plot(line.get_xydata(), running_mean, label='10-element-running-mean' + line.get_label())
        new_line = ax.plot(line.get_xdata(), running_mean, line.get_color())


ax.legend(loc='upper right', fontsize='x-large', ncols=3)
ax.grid(True)
fig.show()
plt.pause(1000)

