import json
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from six import unichr

with open('../summary_data/best_policies_avg_total_return.json', 'r') as file:
    d = json.loads(file.read())

grouped_dic = defaultdict(list)
for key, value in d.items():
    key_parts = key.split('_')
    agent_count = key_parts[0]
    function = str.join(' ', key_parts[2:]).title()
    grouped_dic[agent_count].append({'function': function,
                                     'value': value})
    # final_dic = [{key: value} for key, value in

final_list = defaultdict(list)
for dic in sorted(grouped_dic.items(), key=lambda x: int(x[0]), reverse=False)[1:]:
    for i in dic[1]:
        final_list[i['function']].append(i['value'])



x = np.arange(len(list(grouped_dic.items())[1:]))
width = 1 / (len(final_list)+1)
multiplier = 0

max_value = 0
fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')

for function, values in final_list.items():
    offset = multiplier * width
    rects = ax.bar(x + offset, [-10*value for value in values], width, label=function)
    ax.bar_label(rects, padding=3)
    max_value = max(max_value, max([-10 * item for item in values]))
    multiplier += 1

ax.set_ylabel('Συνολική μέση ανταμοιβή μίας ημέρας ({})'.format(u'\u20AC'))
ax.set_title('Μέση ανταμοιβή μίας ημέρας στο δείγμα αξιολόγησης για τους διάφορους μηχανισμούς τιμολόγησης'.title())
ax.set_xticks(x + width * 1.5, [key + ' Agents' for key, item
                                in sorted(grouped_dic.items(), key=lambda x: int(x[0]), reverse=False)[1:]])
ax.legend(loc='upper left', fontsize=20, ncols=4)
ax.set_ylim(0,max_value + 100 )

plt.pause(1000)
fig.show()