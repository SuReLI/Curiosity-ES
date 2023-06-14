from env.GymMaze.CMaze import CMaze
import sys
import json


# replay bccord
path=sys.argv[1]
env_name=path.split('/')[-2]
with open(path) as json_file:
    data = json.load(json_file)

archive_coord=data['archive_coord']

# env
env=CMaze(filename=env_name)
env.render(behaviours=archive_coord, freeze=True)
# print(len(archive_coord))