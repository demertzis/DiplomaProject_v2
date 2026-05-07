import csv
import itertools
import json
import os
from datetime import datetime
from sshkeyboard import listen_keyboard, stop_listening
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import tkinter as tk

import config

# plot_actions = True
plot_actions = False

directory = config.PLOTS_FOLDER
index = 0
dir_list = []
for item in os.listdir(directory):
    # Check if the current item is a directorypip
    if os.path.isdir(os.path.join(directory, item)):
        # Process the directory
        dir_list.append('/'.join([directory, item]))
        dir_list.sort(key=lambda x: int("".join(item for item in list(filter(str.isdigit, x)))))

num_of_agents_list = ["".join(list(filter(str.isdigit, i))) for i in dir_list]

dates_array = []
with open(config.EVAL_PRICES_FILE, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for i in range(config.NUM_EVAL_EPISODES):
        dates_array.append(datetime.strptime(next(csv_reader)[0], '%Y-%m-%d %H:%M:%S').date())
        for _ in range(23):
            next(csv_reader)

while True:
    print('Available number of agents: {}'.format(str(num_of_agents_list)))
    num_of_agents = input('Choose the number of Agents by typing the number and press Enter\n')
    if num_of_agents not in [i for i in num_of_agents_list]:
        print('Input typed ({}) does not correspond to a valid number. Try again'.format(num_of_agents))
        continue

    chosen_dir = dir_list[num_of_agents_list.index(num_of_agents)]

    print('{} Agents selected'.format(num_of_agents))

    listdir = os.listdir(chosen_dir)
    func_list = [str(i) for i in range(1, len(listdir) + 1)]

    while True:
        for index, item in enumerate(listdir):
            print('{} -> {}'.format(index + 1, os.path.splitext(os.path.basename(item))[0]))
        chosen_functions = input('Choose one or more of the available reward functions by typing the corresponding '
                                 'numbers from the list above separated by commas and press Enter or press B and then '
                                 'Enter to choose number of agents\n')

        chosen_functions_list = [i.strip() for i in chosen_functions.split(sep=',')]
        if chosen_functions == 'B':
            break
        elif not all(i.strip() in func_list for i in chosen_functions_list):
            print('Input typed ({}) does not correspond to a valid reward functions. Try again\n'.format(chosen_functions))
            continue


        chosen_reward_function_names = [os.path.splitext(os.path.basename(listdir[index]))[0] \
                                        for index in [func_list.index(i) \
                                                      for i in set(chosen_functions.split(sep=','))]]
        print('\n'.join(['Reward functions selected:'] + chosen_reward_function_names))
        # print('"{}" reward functions selected:\n'.format(' '.join(os.path.splitext(os.path.basename(listdir[index]))[0] \
        #                                                  for index in [func_list.index(i) \
        #                                                                for i in set(chosen_functions_list)])))
        # print('Press Enter to show the first day of evaluation. By pressing Enter you go to the next day cycling to the'
        #       'first at the end. Press B and then Enter to go back to choosing the function')

        np_array_list = []
        for i in chosen_reward_function_names:
            # with open('/'.join([chosen_dir, i + '.json']), 'r') as json_file:
            with open('/'.join([chosen_dir, i, 'actions.json' if plot_actions else 'rewards.json']), 'r') as json_file:
            # with open('/'.join([chosen_dir, i, 'rewards.json']), 'r') as json_file:
                np_array_list.append({'reward_name': i, 'evaluation_data': np.array(json.load(json_file))})

        if any(arr['evaluation_data'].shape[0] != np_array_list[0]['evaluation_data'].shape[0] for arr in np_array_list):
            raise Exception('Some of the reward functinos does not have data for every day of the evaluation set')

        max_value = np.max(max([np.sum(i['evaluation_data'], axis=-1) for i in np_array_list], key=lambda i: np.max(i)))
        min_value = np.min(min([np.sum(i['evaluation_data'], axis=-1) for i in np_array_list], key=lambda i: np.min(i)))

        def onkeypress(event):
            # global toggle

            # toggle = not toggle
            # fig.clear()
            global day
            if event.key == 'left' or event.key == 'right':
                # event.canvas.figure.clear()
                index_change = 1 if event.key == 'right' else -1
                day = (day + index_change) % np_array_list[0]['evaluation_data'].shape[0]
                ax = event.canvas.figure.gca()
                # for data in np_array_list:
                for line, data in zip(lines_array, np_array_list):
                    line.set(ydata=np.sum(data['evaluation_data'][day], axis=-1))
                #     ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
                # print(type(lines_array[i]))
                # line.set_ydata(np.sum(data['evaluation_data'][day], axis=-1))
                # event.canvas.figure.gca().set_title('Reward Graph for day '
                #                                     '{}'.format(dates_array[day].strftime('%d-%m-%Y')))  # Add a title to the axes.
                # for data in np_array_list:
                #     ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
                # ax.set_xlabel('Time of the Day')  # Add an x-label to the axes.
                # ax.set_ylabel('Load x Price ')  # Add a y-label to the axes.
                # ax.set_title('Reward Graph for {}'.format(dates_array[day].strftime('%d-%m-%Y')))  # Add a title to the axes.
                # ax.grid(True)
                # lines_array = []
                # ax.get_legend().remove()
                ax.legend().remove()
                ax.legend(fontsize='large')  # Add a legend.
                ax.set_title('Reward Graph for day {}'.format(dates_array[day].strftime('%d-%m-%Y')))

                fig.canvas.draw()
            else:
                return

            # if toggle:
                # event.canvas.figure.gca().plot(Data1)
            # if event.key == 'right':
            #     index = (index + 1) % np_array_list.shape[0]
            #     for data in np_array_list:
            #         event.canvas.figure.gca().plot(x, np.sum(data['evaluation_data'][index], axis=-1), label=data['reward_name'])
            #
            # else:
            #     index = (index - 1) % np_array_list.shape[0]
            #     for data in np_array_list:
            #         event.canvas.figure.gca().plot(x, np.sum(data['evaluation_data'][index], axis=-1), label=data['reward_name'])
            #
            # event.canvas.draw()

        day = 0
        # plot_buffer = cycle(list(range(numpy_array.shape[0])))
        fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
        x = np.arange(0, 24, dtype=int)
        # Plot some data on the axes.

        ax.set_xlabel('Time of the Day')  # Add an x-label to the axes.
        ax.set_ylabel('Load x Price ')  # Add a y-label to the axes.
        ax.set_ylim([min_value, max_value])
        ax.set_title('Reward Graph for {}'.format(dates_array[day].strftime('%d-%m-%Y')))  # Add a title to the axes.
        ax.grid(True)
        ax.legend(fontsize='large')  # Add a legend.
        lines_array = []
        for data in np_array_list:
            # ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
            lines_array.append(ax.plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])[0])
        fig.canvas.mpl_connect('key_press_event', onkeypress)

        fig.show()

