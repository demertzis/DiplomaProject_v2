import glob

import numpy as np
import json

import config

directory = config.PLOTS_FOLDER
individual = True
# individual = False

complete_array_list = []
dir_list = []
for file in glob.glob('../' + config.PLOTS_FOLDER + '/**/' + '*.json', recursive=True):
    with open(file, "r") as json_file:
        path = file.split('/')[1:]
        number_of_agents = "".join(item for item in list(filter(str.isdigit, path[1])))
        reward_name = " ".join(filter(lambda x: False if x == "new" or x == "reward" else True,
                                  path[2].split('_')))
        type = path[3].split('.')[0]
        data = np.array(json.load(json_file))
        complete_array_list.append({'number_of_agents': number_of_agents,
                                    'reward_name': reward_name,
                                    'type': type,
                                    'evaluation_data': data})

agents = '3'
# agents = '6'
# agents = '15'

# type = 'actions'
type = 'rewards'

comparison_mechanism_1 = 'trained single agent'
# comparison_mechanism_2 = ('halfway')
# comparison_mechanism_2 = ('buyers biased')
# comparison_mechanism_2 = ('sellers biased')
comparison_mechanism_2 = ('proportional punishing')

final_data = list(filter(lambda e: e['number_of_agents'] == agents and \
                                   e['reward_name'] == comparison_mechanism_1 and \
                                   e['type'] == type,
                         complete_array_list))[0]['evaluation_data']
final_data_2 = list(filter(lambda e: e['number_of_agents'] == agents and \
                                     e['reward_name'] == comparison_mechanism_2 and \
                                     e['type'] == type,
                           complete_array_list))[0]['evaluation_data']

np_array = np.array([t[0] - t[1] for t in zip(final_data_2, final_data)])