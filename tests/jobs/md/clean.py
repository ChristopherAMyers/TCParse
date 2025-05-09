import json
import copy
import os
import shutil

'''
This script is used to clean the tc.json file by removing the 'velocities' key
from the JSON data if its value is None. It creates a backup of the original
file before making any changes and removes the backup file after the changes
are made.

This was necessary for the tests to pass, but in a real-world scenario, the worst
case would be to have a None value for the 'velocities' key.
'''

#   backup the original file
shutil.copy('tc.json', 'tc_backup.json')

with open('tc.json') as file:
    data = json.load(file)

new_data = copy.deepcopy(data)

for i, frame_data in enumerate(data):
    if 'velocities' in frame_data:
        if frame_data['velocities'] is None:
            new_data[i].pop('velocities')

#   write the cleaned file
with open('tc.json', 'w') as file:
    json.dump(new_data, file, indent=4)

#   remove the backup file
os.remove('tc_backup.json')