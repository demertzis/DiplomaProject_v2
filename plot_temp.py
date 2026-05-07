import csv
import itertools
import json
import os
import glob
from datetime import datetime
from tkinter import Radiobutton

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from matplotlib.container import BarContainer

import config
plot_actions = False

show_prices = True
# show_prices = False


directory = config.PLOTS_FOLDER
# index = 0
complete_array_list = []
dir_list = []
for file in glob.glob(config.PLOTS_FOLDER + '/**/' + '*.json', recursive=True):
    with open(file, "r") as json_file:
        path = file.split('/')
        number_of_agents = "".join(item for item in list(filter(str.isdigit, path[1])))
        reward_name = " ".join(filter(lambda x: False if x == "new" or x == "reward" else True,
                                  path[2].split('_')))
        type = path[3].split('.')[0]
        data = np.array(json.load(json_file))
        complete_array_list.append({'number_of_agents': number_of_agents,
                                    'reward_name': reward_name,
                                    'type': type,
                                    'evaluation_data': data})
dates_array = []
prices_array = []
with open(config.EVAL_PRICES_FILE, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for i in range(config.NUM_EVAL_EPISODES):
        line = next(csv_reader)
        dates_array.append(datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S').date())
        prices_array.append([[float(line[1]), float(line[2])]])
        for _ in range(23):
            line = next(csv_reader)
            prices_array[-1].append([float(line[1]), float(line[2])])
    prices_array = np.array(prices_array)
# while True:
#     print('Available number of agents: {}'.format(str(num_of_agents_list)))
#     num_of_agents = input('Choose the number of Agents by typing the number and press Enter\n')
#     if num_of_agents not in [i for i in num_of_agents_list]:
#         print('Input typed ({}) does not correspond to a valid number. Try again'.format(num_of_agents))
#         continue
#
#     chosen_dir = dir_list[num_of_agents_list.index(num_of_agents)]
#
#     print('{} Agents selected'.format(num_of_agents))
#
#     listdir = os.listdir(chosen_dir)
#     func_list = [str(i) for i in range(1, len(listdir) + 1)]
#
#     while True:
#         for index, item in enumerate(listdir):
#             print('{} -> {}'.format(index + 1, os.path.splitext(os.path.basename(item))[0]))
#         chosen_functions = input('Choose one or more of the available reward functions by typing the corresponding '
#                                  'numbers from the list above separated by commas and press Enter or press B and then '
#                                  'Enter to choose number of agents\n')
#
#         chosen_functions_list = [i.strip() for i in chosen_functions.split(sep=',')]
#         if chosen_functions == 'B':
#             break
#         elif not all(i.strip() in func_list for i in chosen_functions_list):
#             print('Input typed ({}) does not correspond to a valid reward functions. Try again\n'.format(chosen_functions))
#             continue
#
#
#         chosen_reward_function_names = [os.path.splitext(os.path.basename(listdir[index]))[0] \
#                                         for index in [func_list.index(i) \
#                                                       for i in set(chosen_functions.split(sep=','))]]
#         print('\n'.join(['Reward functions selected:'] + chosen_reward_function_names))
#         # print('"{}" reward functions selected:\n'.format(' '.join(os.path.splitext(os.path.basename(listdir[index]))[0] \
#         #                                                  for index in [func_list.index(i) \
#         #                                                                for i in set(chosen_functions_list)])))
#         # print('Press Enter to show the first day of evaluation. By pressing Enter you go to the next day cycling to the'
#         #       'first at the end. Press B and then Enter to go back to choosing the function')
#
#         np_array_list = []
#         for i in chosen_reward_function_names:
#             # with open('/'.join([chosen_dir, i + '.json']), 'r') as json_file:
#             with open('/'.join([chosen_dir, i, 'actions.json' if plot_actions else 'rewards.json']), 'r') as json_file:
#             # with open('/'.join([chosen_dir, i, 'rewards.json']), 'r') as json_file:
#                 np_array_list.append({'reward_name': i, 'evaluation_data': np.array(json.load(json_file))})
#
#         if any(arr['evaluation_data'].shape[0] != np_array_list[0]['evaluation_data'].shape[0] for arr in np_array_list):
#             raise Exception('Some of the reward functinos does not have data for every day of the evaluation set')
#
#         max_value = np.max(max([np.sum(i['evaluation_data'], axis=-1) for i in np_array_list], key=lambda i: np.max(i)))
#         min_value = np.min(min([np.sum(i['evaluation_data'], axis=-1) for i in np_array_list], key=lambda i: np.min(i)))
#
#         def onkeypress(event):
#             # global toggle
#
#             # toggle = not toggle
#             # fig.clear()
#             global day
#             if event == 'left' or event == 'right':
#                 # event.canvas.figure.clear()
#                 index_change = 1 if event == 'right' else -1
#                 day = (day + index_change) % np_array_list[0]['evaluation_data'].shape[0]
#                 ax = event.canvas.figure.gca()
#                 # for data in np_array_list:
#                 for line, data in zip(lines_array, np_array_list):
#                     line.set(ydata=np.sum(data['evaluation_data'][day], axis=-1))
#                 #     ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
#                 # print(type(lines_array[i]))
#                 # line.set_ydata(np.sum(data['evaluation_data'][day], axis=-1))
#                 # event.canvas.figure.gca().set_title('Reward Graph for day '
#                 #                                     '{}'.format(dates_array[day].strftime('%d-%m-%Y')))  # Add a title to the axes.
#                 # for data in np_array_list:
#                 #     ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
#                 # ax.set_xlabel('Time of the Day')  # Add an x-label to the axes.
#                 # ax.set_ylabel('Load x Price ')  # Add a y-label to the axes.
#                 # ax.set_title('Reward Graph for {}'.format(dates_array[day].strftime('%d-%m-%Y')))  # Add a title to the axes.
#                 # ax.grid(True)
#                 # lines_array = []
#                 # ax.get_legend().remove()
#                 ax.legend().remove()
#                 ax.legend(fontsize='large')  # Add a legend.
#                 ax.set_title('Reward Graph for day {}'.format(dates_array[day].strftime('%d-%m-%Y')))
#
#                 fig.canvas.draw()
#             else:
#                 return
#
#             # if toggle:
#                 # event.canvas.figure.gca().plot(Data1)
#             # if event == 'right':
#             #     index = (index + 1) % np_array_list.shape[0]
#             #     for data in np_array_list:
#             #         event.canvas.figure.gca().plot(x, np.sum(data['evaluation_data'][index], axis=-1), label=data['reward_name'])
#             #
#             # else:
#             #     index = (index - 1) % np_array_list.shape[0]
#             #     for data in np_array_list:
#             #         event.canvas.figure.gca().plot(x, np.sum(data['evaluation_data'][index], axis=-1), label=data['reward_name'])
#             #
#             # event.canvas.draw()
#
#         day = 0
#         # plot_buffer = cycle(list(range(numpy_array.shape[0])))
#         fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
#         x = np.arange(0, 24, dtype=int)
#         # Plot some data on the axes.
#
#         ax.set_xlabel('Time of the Day')  # Add an x-label to the axes.
#         ax.set_ylabel('Load x Price ')  # Add a y-label to the axes.
#         ax.set_ylim([min_value, max_value])
#         ax.set_title('Reward Graph for {}'.format(dates_array[day].strftime('%d-%m-%Y')))  # Add a title to the axes.
#         ax.grid(True)
#         ax.legend(fontsize='large')  # Add a legend.
#         lines_array = []
#         for data in np_array_list:
#             # ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
#             lines_array.append(ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])[0])
#         fig.canvas.mpl_connect('key_press_event', onkeypress)
#
#         fig.show()
print("Use left/right arrow keys to swith through days,"
      " up/down keys to switch between numbers of stations,"
      " press 'm' to toggle between rewards and actions"
      "press 'i' to toggle between individual agents or the sum"

      )

def get_line_label(array_dict, agent = None):
    return "-".join((array_dict['type'],
                     array_dict['number_of_agents'],
                     "stations",
                     array_dict['reward_name'],
                     "station" + str(agent + 1) if isinstance(agent,int) else "Summed Up" ))

day = 0
np_array_list = []

def plot_data(number_of_agents_chosen, reward_functions_chosen, toggle, individual):
        # chosen_dir = dir_list[num_of_agents_list.index(number_of_agents_chosen)]
        # chosen_dirs = [dir if any([i in dir for i in number_of_agents_chosen]) else None for dir in dir_list]
        global np_array_list
        np_array_list = list(filter(lambda dic: dic['reward_name'] in reward_functions_chosen and
                                                dic['number_of_agents'] in number_of_agents_chosen and
                                                dic['type'] == toggle,
                                    complete_array_list))

        if any(arr['evaluation_data'].shape[0] != np_array_list[0]['evaluation_data'].shape[0] for arr in np_array_list):
            raise Exception('Some of the reward functions do not have data for every day of the evaluation set')

        individual_min_value = np.min([np.min(i['evaluation_data']) for i in np_array_list])
        individual_max_value = np.max([np.max(i['evaluation_data']) for i in np_array_list])

        if toggle == 'rewards':
            for arr in np_array_list:
                arr.update({'evaluation_data': arr['evaluation_data'] * 10.0})
        max_value = np.max(max([np.sum(i['evaluation_data'], axis=-1) for i in np_array_list], key=lambda i: np.max(i)))
        min_value = np.min(min([np.sum(i['evaluation_data'], axis=-1) for i in np_array_list], key=lambda i: np.min(i)))

        global day
        day=0
        global x
        x = np.arange(0, 24, dtype=int)
        global fig

        if show_prices:
            fig, axes = plt.subplots(nrows=2, figsize=(7, 5),sharex=True, gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
            ax = axes[0]
            axes[1].set_ylim([np.min(prices_array), np.max(prices_array)])
            axes[1].grid(True)
            axes[1].set_xlabel('Time of the Day')  # Add an x-label to the axes.
            axes[1].set_ylabel('Price (\u20AC / MWh)')  # Add a y-label to the axes.
            bars_dam = axes[1].bar(x, [i[0] for i in prices_array[day]], 0.5, align='center', facecolor='b', label='Day Ahead Prices')
            bars_id = axes[1].bar(x, [i[1] for i in prices_array[day]], 0.5, align='center', facecolor='y', label='Intra Day Prices')
            for bar in bars_dam:
                bar.set_zorder(-bar.get_height())
            for bar in bars_id:
                bar.set_zorder(-bar.get_height())
            axes[1].legend(fontsize='large')
            # for i, x_i in enumerate(x):
            #     axes[1].bar([x_i] + 0.25, [prices_array[day][0]], 0.5, align='center', zorder=-prices_array[day[0]], facecolor='b')
            #     axes[1].bar([x_i] + 0.25, [prices_array[day][1]], 0.5, align='center',  zorder=-prices_array[day[1]],facecolor='y')
            # axes[1].bar(x + 0.25, [i[0] for i in prices_array[day]], 0.5)
        else:
            fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')

        # Plot some data on the axes.

        ax.set_xlabel('Time of the Day')  # Add an x-label to the axes.
        if toggle == 'actions':
            ax.set_ylabel('Load (KWh) ')  # Add a y-label to the axes.
        else:
            ax.set_ylabel('Reward (\u20AC)')  # Add a y-label to the axes.
        ax.set_ylim([individual_min_value, individual_max_value] if individual else [min_value, max_value])
        ax.set_title(create_graph_title() + " at " + dates_array[day].strftime('%d-%m-%Y'))  # Add a title to the axes.
        ax.grid(True)
        # ax.legend(fontsize='large')  # Add a legend.
        # lines_array = []
        for array in np_array_list:
            # ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
            if individual:
                for agent in range(array["evaluation_data"].shape[-1]):
                    if array['number_of_agents'] == '1':
                        if 'offset' in array['reward_name']:
                            fmt = 'k:'
                        else:
                            fmt = 'k--'
                    else:
                        fmt = ''
                    line = ax.plot(x, array["evaluation_data"][day, :, agent], fmt, label=get_line_label(array, agent),)
                    array.update(dict(agent=agent))
                    line[0].metadata={key: value for key, value in array.items() if key not in ("evaluation_data")}#Custom property
                    pass
                    # lines_array.append(
                    #                    )
            else:
                if array['number_of_agents'] == '1':
                    if 'offset' in array['reward_name']:
                        fmt = 'k:'

                    else:
                        fmt = 'k--'
                else:
                    fmt = ''
                line = ax.plot(x,
                               np.sum(array["evaluation_data"][day], axis=-1),
                               fmt,
                               label=get_line_label(array))
                line[0].metadata = {key: value for key, value in array.items() if key not in ("evaluation_data")}  # Custom property
                # lines_array.append(ax.plot(x, np.sum(array["evaluation_data"][day], axis=-1),
                #                            label=get_line_label(array)
                #                            )
                #                    )
        if len(np_array_list) == 1 and individual and np_array_list[0]['number_of_agents'] != '1':
        # if all((array['reward_name'] == np_array_list[0]['reward_name']) and \
        #        (array['number_of_agents'] == np_array_list[0]['number_of_agents']) \
        #        for array in np_array_list) and all(array.items() for array in np_array_list):
            ax.bar(x, np.sum(np_array_list[0]["evaluation_data"][day], axis=-1), width=1, edgecolor="white", alpha=0.3, linewidth=0.7)

            # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
            #        ylim=(0, 8), yticks=np.arange(1, 8))
            # pass
            ax.set_ylim([min_value, max_value])
        # fig.canvas.mpl_connect('key_press_event', onkeypress)
        # ax.legend().remove()
        ax.legend(fontsize='large')  # Add a legend.
        fig.show()

def onkeypress(event, individual):
    # global toggle

    # toggle = not toggle
    # fig.clear()
    global day
    # global individual
    # global toggle
    if event == 'left' or event == 'right':
        index_change = 1 if event == 'right' else -1
        day = (day + index_change) % np_array_list[0]["evaluation_data"].shape[0]
        # axes = fig.gca()
        axes = fig.get_axes()
        ax = axes[0]
        if show_prices:
            # for container in axes[1].containers:
            #     container.remove()
            containers = axes[1].containers
            bars_dam = next(filter(lambda i: i.get_label() == 'Day Ahead Prices',containers))
            bars_id = next(filter(lambda i: i.get_label() == 'Intra Day Prices',containers))
            # bars_dam.datavalues = [i[0] for i in prices_array[day]]
            # bars_id.datavalues = [i[1] for i in prices_array[day]]
            # bars_dam = axes[1].bar(x, [i[0] for i in prices_array[day]], 0.5, align='center', facecolor='b', label='Day Ahead Prices')
            # bars_id = axes[1].bar(x, [i[1] for i in prices_array[day]], 0.5, align='center', facecolor='y', label='Intra Day Prices')
            for i, bar in enumerate(bars_dam):
                bar.set_height(prices_array[day, i, 0])
                bar.set_zorder(-bar.get_height())
            for i, bar in enumerate(bars_id):
                bar.set_height(prices_array[day, i, 1])
                bar.set_zorder(-bar.get_height())
            axes[1].legend().remove()
            axes[1].legend(fontsize='large')
        lines = list(ax.get_lines())
        # ax.cla()
        for line in lines:
            new_data = [array["evaluation_data"] for array  in np_array_list if len(array.items() & line.metadata.items()) > 2][0]
            if line.metadata.get('agent') is not None:
                new_data = new_data[day,:,line.metadata['agent']]
            else:
                new_data = np.sum(new_data[day], axis=-1)
            line.set(ydata=new_data)
        bars = [i for i in ax.containers if isinstance(i, BarContainer)]
        if len(bars) == 1:
            # bars[0].set_height = np.sum(np_array_list[0]["evaluation_data"][day], axis=-1)
            for i, bar in enumerate(list(bars[0])): bar.set_height(np.sum(np_array_list[0]["evaluation_data"][day][i], ))


        # for line in np_array_list:
        #     # ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
        #     if individual:
        #         for agent in range(line['evaluation_data'].shape[-1]):
        #                 ax.plot(x, line['evaluation_data'][day, :, agent], label=get_line_label(line, agent))
        #     else:
        #         ax.plot(x, np.sum(line['evaluation_data'][day], axis=-1), label=get_line_label(line))

        # for line, data in zip(lines_array, np_array_list):
        #     line.set(ydata=np.sum(data['evaluation_data'][day], axis=-1))
        ax.legend().remove()
        ax.legend(fontsize='large')  # Add a legend.
        ax.set_title(create_graph_title() + " at " + dates_array[day].strftime('%d-%m-%Y'))
        fig.canvas.draw()

    else:
        return

class Checkbar(tk.Frame):
    def __init__(self, parent=None, picks=[], side=tk.LEFT, anchor=tk.W):
        tk.Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = tk.StringVar()
            chk = tk.Checkbutton(self, text=pick, variable=var, onvalue=pick, offvalue="")
            chk.pack(side=side, anchor=anchor, expand=tk.YES)
            self.vars.append(var)

    def state(self):
        return map((lambda var: var.get()), self.vars)


if __name__ == '__main__':
    root = tk.Tk()
    reward_checkbar = Checkbar(root, ['pretrained policy',
                                      'trained single agent',
                                      'halfway',
                                      'buyers biased',
                                      'sellers biased',
                                      'proportional punishing'])
    agent_count_checkbar = Checkbar(root, ['1', '3', '6', '15'])

    toggle = tk.StringVar()
    toggle_actions = Radiobutton(root, text="Actions", variable=toggle, value="actions", font=('utopia', 40))

    toggle_rewards = Radiobutton(root, text="Rewards", variable=toggle, value="rewards")


    individual = tk.BooleanVar()
    toggle_individual = Radiobutton(root, text="individual agents", variable=individual, value=True)
    toggle_sum = Radiobutton(root, text="sum_of_agents", variable=individual, value=False)

    # individual_agents = Checkbar(root, ['1', '3', '6', '15'])

    reward_checkbar.pack(side=tk.TOP, fill=tk.X)
    agent_count_checkbar.pack(side=tk.LEFT)
    toggle_rewards.pack(side=tk.BOTTOM, anchor="w")
    toggle_actions.pack(side=tk.BOTTOM, anchor="w")
    toggle_individual.pack(side=tk.BOTTOM, anchor="w")
    toggle_sum.pack(side=tk.BOTTOM, anchor="w")
    reward_checkbar.config(relief=tk.GROOVE, bd=2)


    # def allstates():
    #     print(list(reward_checkbar.state()), list(agent_count_checkbar.state()))
    def create_graph_title():
        return (("Actions" if toggle == 'actions' else "Rewards") + " for " +
                ','.join(list(agent_count_checkbar.state())) +
                " Stations"
                " with Reward Functions " +
                ','.join(list(reward_checkbar.state())))


    tk.Button(root, text='Quit', command=root.quit).pack(side=tk.RIGHT)
    def offset_addition(list):
        if 'trained single agent' in list:
            list.append('trained single agent offset')
        if 'pretrained policy' in list:
            list.append('pretrained policy offset')
        return list
    tk.Button(root, text='Peek', command=lambda: plot_data(list(agent_count_checkbar.state()),
                                                           offset_addition(list(reward_checkbar.state())),
                                                           toggle.get(),
                                                           individual.get())).pack(side=tk.RIGHT)
    tk.Button(root, text='Next Day', command=lambda: onkeypress(event='right', individual=individual.get())).pack(side=tk.RIGHT)
    tk.Button(root, text='Previous Day', command=lambda: onkeypress(event='left', individual=individual.get())).pack(side=tk.RIGHT)
    root.resizable(width=1, height=1)
    root.mainloop()


# root = tk.Tk()
# myapp = App(root)any([i in dir for i in number_of_agents_chosen])
# myapp.mainloop()


