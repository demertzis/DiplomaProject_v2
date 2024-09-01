import csv
import json
import os
from datetime import datetime
from sshkeyboard import listen_keyboard, stop_listening
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

import config

directory = 'plots'
index = 0
dir_list = []
for item in os.listdir(directory):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(directory, item)):
        # Process the directory
        dir_list.append('/'.join([directory, item]))
        dir_list.sort(key=lambda x: int("".join(item for item in list(filter(str.isdigit, x)))))

num_of_agents_list = ["".join(list(filter(str.isdigit, i))) for i in dir_list]

dates_array = []
energy_array = np.zeros(shape=[config.NUM_EVAL_EPISODES, 24, 2])


with open('data/randomized_data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for i in range(config.NUM_EVAL_EPISODES):
        line = next(csv_reader)
        dates_array.append(datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S').date())
        energy_array[i][0][0], energy_array[i][0][1] = line[1], line[2]
        for j in range(23):
            new_line = next(csv_reader)
            energy_array[i][j + 1][0], energy_array[i][j + 1][1] = new_line[1], new_line[2]

while True:
    print('Available number of agents: {}'.format(str(num_of_agents_list)))
    num_of_agents = input('Choose the number of Agents by typing the number and press Enter\n')
    if num_of_agents not in num_of_agents_list:
        print('Input typed ({}) does not correspond to a valid number. Try again'.format(num_of_agents))
        continue

    chosen_dir = dir_list[num_of_agents_list.index(num_of_agents)]

    print('{} Agents selected'.format(num_of_agents))

    listdir = os.listdir(chosen_dir)
    func_list = [str(i) for i in range(1, len(listdir) + 1)]

    while True:
        criteria_list = ['Maximum Average Reward', 'Maximum Average Action Deviation']
        corresponding_criteria_indexes = list(map(lambda i: str(i), range(1, len(criteria_list) + 1)))
        for index, criteria in zip(corresponding_criteria_indexes, criteria_list):
            print('{}: {}'.format(index, criteria))

        chosen_criteria = input('Choose criteria to compare the different reward functions against each other '
                                'by typing the number corresponding to each criteria from the above list of type B '
                                'to get back to choosing number of agents\n').strip()
        if chosen_criteria == 'B':
            break
        elif chosen_criteria not in corresponding_criteria_indexes:
            print('Input typed ({}) does not correspond to a valid criteria according to'
                  ' the above list. Try again'.format(chosen_criteria))
            continue
        numpy_reward_list = []
        numpy_action_list = []

        for func_dirs in os.listdir(chosen_dir):
            func_dir_path = os.path.join(chosen_dir, func_dirs)
            if os.path.isfile(func_dir_path):
                continue
            func_name = os.path.basename(func_dirs)
            with open(os.path.join(func_dir_path, 'rewards.json'), 'r') as json_file:
                numpy_reward_list.append({'reward_name': func_name, 'evaluation_data': np.array(json.load(json_file))})
            with open(os.path.join(func_dir_path, 'actions.json'), 'r') as json_file:
                numpy_action_list.append({'reward_name': func_name, 'evaluation_data': np.array(json.load(json_file))})
        sorted_indexes = sorted(range(len(numpy_reward_list)),
                                key=lambda i: np.average(np.sum(numpy_reward_list[i]['evaluation_data'],
                                                                axis=(-1, -2))),
                                reverse=True)
        sorted_numpy_list = [numpy_reward_list[id] for id in sorted_indexes]
        sorted_numpy_action_list = [numpy_action_list[id] for id in sorted_indexes]
        match chosen_criteria:
            case '1':
                print('Bellow the different reward functions are listed in descending order on the total reward of all '
                      'agents summed for a day on the evaluation data (20 days)')
                print('\n'.join([numpy_reward_list[index]['reward_name'] + \
                                 ': ' + \
                                 str(np.average(np.sum(numpy_reward_list[index]['evaluation_data'], axis=(-1, -2)))) for index in sorted_indexes]))
                sorted_days_indexes = sorted(range(config.NUM_EVAL_EPISODES),
                                             key=lambda day: np.sum(sorted_numpy_list[0]['evaluation_data'][day]) - \
                                                            max([np.sum(func['evaluation_data'][day]) \
                                                                 for func in sorted_numpy_list[1:]]),
                                             reverse=True)
            case '2':
                print('Bellow the different reward functions are listed in descending order on the total action '
                      'deviation of all agents summed for a day on the evaluation data (20 days)')
                print('\n'.join([numpy_reward_list[index]['reward_name'] + \
                                 ': ' + \
                                 str(np.average(np.sum(numpy_reward_list[index]['evaluation_data'], axis=(-1, -2)))) for index in sorted_indexes]))
                average_action_rest = np.average(np.stack([i['evaluation_data'] for i in sorted_numpy_action_list[1:]],
                                                          axis=-1),
                                                 axis=-1)
                sorted_days_indexes = sorted(range(config.NUM_EVAL_EPISODES),
                                             key=lambda day: np.sum(np.abs(sorted_numpy_action_list[0]['evaluation_data'][day] - \
                                                                           average_action_rest[day])),
                                             reverse=True)
                pass

        max_reward_value = np.max(
            max([np.sum(i['evaluation_data'], axis=-1) for i in numpy_reward_list], key=lambda i: np.max(i)))
        min_reward_value = np.min(
            min([np.sum(i['evaluation_data'], axis=-1) for i in numpy_reward_list], key=lambda i: np.min(i)))

        max_action_value = np.max(
            max([i['evaluation_data'] for i in numpy_action_list], key=lambda i: np.max(i)))
        min_action_value = np.min(
            min([i['evaluation_data'] for i in numpy_action_list], key=lambda i: np.min(i)))

        def onkeypress(event):
            global day_index
            global agent_index
            # global axs
            if event.key == 'left' or event.key == 'right':
                agent_index = 0
                index_change = 1 if event.key == 'right' else -1
                day_index = (day_index + index_change) % numpy_reward_list[0]['evaluation_data'].shape[0]
                day = sorted_days_indexes[day_index]

                for data, line in zip(sorted_numpy_list, reward_lines):
                    line.set(ydata=np.sum(data['evaluation_data'][day], axis=-1))

                for data, line in zip(sorted_numpy_action_list, action_lines):
                    line.set(ydata=data['evaluation_data'][day, :, agent_index])

                    # line = list(filter(lambda x: x.get_label() == data['reward_name'], reward_lines))[0]
                    # line.set(ydata=np.sum(data['evaluation_data'][day], axis=-1))

                # for data in sorted_numpy_list:
                #     line = list(filter(lambda x: x.get_label() == data['reward_name'], reward_lines))[0]
                #     line.set(ydata=np.sum(data['evaluation_data'][day], axis=-1))

                # axs[0].legend().remove()
                # axs[0].legend(fontsize='large')  # Add a legend.
                axs[0].set_title('Rewards for day {}, Position in descending order of {}: {}'
                                 ''.format(dates_array[day].strftime('%d-%m-%Y'),
                                           # day,
                                           criteria_list[int(chosen_criteria) - 1],
                                           day_index + 1))

                # axs[1].legend().remove()
                # axs[1].legend(fontsize='large')  # Add a legend.
                axs[1].set_title('Action for agent {}'.format(agent_index + 1))

                for id, line in enumerate(prices_lines):
                    line.set(ydata=energy_array[day_index, :, id])
                # if prices_lines[0].get_label() == 'Intra-Day Prices':
                #     index_complement = 0
                # else:
                #     index_complement = 1
                # intra_prices_line = prices_lines[0 + index_complement]
                # day_ahead_prices_line = prices_lines[1 - index_complement]
                # intra_prices_line.set(ydata=energy_array[day_index, :, 0])
                # day_ahead_prices_line.set(ydata=energy_array[day_index, :, 1])
                fig.canvas.draw()
            elif event.key == 'up' or event.key == 'down':
                day = sorted_days_indexes[day_index]
                agent_index = (agent_index + 1) % int(num_of_agents)
                for data, line in zip(sorted_numpy_action_list, action_lines):
                    line.set(ydata=data['evaluation_data'][day, :, agent_index])
                axs[1].set_title('Action for agent {}'.format(agent_index + 1))
                fig.canvas.draw()
            else:
                return

        day_index = 0
        agent_index = 0
        # plot_buffer = cycle(list(range(numpy_array.shape[0])))
        fig, axs = plt.subplots(3, 1, figsize=(7, 5), sharex=True, layout='constrained')
        x = np.arange(0, 24, dtype=int)
        # Plot some data on the axes.

        plt.xticks(x, x)

        axs[2].set_xlabel('Time of the Day')  # Add an x-label to the axes.
        axs[0].set_ylabel('Reward = Load x Price ')  # Add a y-label to the axes.
        axs[1].set_ylabel('Action of Chosen Agent (Load in Kwh)')
        axs[0].set_ylim([min_reward_value, max_reward_value])
        axs[1].set_ylim([min_action_value, max_action_value])
        axs[2].set_ylim([np.min(energy_array[:,:,:]),
                         np.max(energy_array[:,:,:])])

        day = sorted_days_indexes[day_index]
        # axs[0].set_title('Reward Graph for {}, {}'.format(dates_array[day].strftime('%d-%m-%Y'), day))  # Add a title to the axes.
        axs[0].set_title('Rewards for day {}, Position in descending order of {}: {}'
                         ''.format(dates_array[day].strftime('%d-%m-%Y'),
                                   # day,
                                   criteria_list[int(chosen_criteria) - 1],
                                   day_index + 1))# Add a title to the axes.

        axs[1].set_title('Action for Agent {}'.format(agent_index + 1))

        for ax in axs:
            ax.grid(True, linestyle='-', linewidth=1, color='gray')

        for data in sorted_numpy_list:
            axs[0].plot(x, np.sum(data['evaluation_data'][day], axis=-1), label=data['reward_name'])
            # axs[1].plot

        # for i in sorted_data_indexes
            # axs[1].plot(x, data['evaluation_data'][day][agent_index], label=data['reward_name'])

        # for data in numpy_action_list:
        #     axs[1].plot(x, data['evaluation_data'][day][agent_index], label=data['reward_name'])

        # for data in
        sorted_name_list = [elem['reward_name'] for elem in sorted_numpy_list]
        reward_lines = sorted(axs[0].get_lines(),
                              key=lambda i: sorted_name_list.index(i.get_label()),
                              reverse=False)
        axs[0].legend(fontsize='large')  # Add a legend.

        for data in sorted_numpy_action_list:
            axs[1].plot(x, data['evaluation_data'][day, :, agent_index], label=data['reward_name'])
        axs[1].legend(fontsize='large')

        action_lines = sorted(axs[1].get_lines(),
                              key=lambda i: sorted_name_list.index(i.get_label()),
                              reverse=False)

        axs[2].plot(energy_array[day_index, :, 0], label='Intra-Day Prices')
        axs[2].plot(energy_array[day_index, :, 1], label='Day-Ahead Prices')
        prices_lines = sorted(axs[2].get_lines(),
                              key=lambda i: int(i.get_label() == 'Intra-Day Prices'),
                              reverse=True)

        axs[2].set_title('Prices')
        axs[2].legend(fontsize='large')
        fig.canvas.mpl_connect('key_press_event', onkeypress)

        fig.show()

