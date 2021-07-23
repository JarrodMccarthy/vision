import os
from os import walk

root = os.getcwd()
path = os.path.join(root, 'data', 'faces', 'no')
files = list()
for _,_,filenames in walk(path):
    files = filenames

for item in files:
    oldnum = item.split('.')[0]
    filetype = item.split('.')[1]
    newnum = int(oldnum)-1500
    newname = str(newnum)+'.'+filetype

    oldpath = os.path.join(path, item)
    newpath = os.path.join(path, newname)

    os.rename(oldpath, newpath)




# import os

# old_file_name = "/home/career_karma/raw_data.csv"
# new_file_name = "/home/career_karma/old_data.csv"

# os.rename(old_file_name, new_file_name)

# print("File renamed!")
