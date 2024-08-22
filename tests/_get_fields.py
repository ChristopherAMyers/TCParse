import os
import glob
import json

def add(x: str):
    if x.isdigit():
        return False
    if len(x) > 1:
        return True

fields = {}
def find_keys(obj):
    global fields
    if isinstance(obj, str):
        return
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if add(k):
                fields[k] = type(v)
            find_keys(v)
    elif isinstance(obj, list):
        for x in obj:
            find_keys(x)


for file_loc in glob.glob('*/*/*.json', recursive=True):
    with open(file_loc) as file:
        data = json.load(file)
    find_keys(data)

required_lines = []
lines = []
for k in sorted(fields):
    type_str = str(fields[k])
    type_str = type_str.split("'")[1]
    type_str = type_str.replace('NoneType', 'None')

    if k in ['atoms', 'geom', 'energy']:
        required_lines.append(f'    {k:30s}: list')
    else:
        lines.append(f'    {k:30s}: Optional[{type_str}] = None')
        
for line in required_lines:
    print(line)
print()
for line in lines:
    print(line)