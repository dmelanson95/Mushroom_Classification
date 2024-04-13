import pandas as pd
import os

os.chdir('C:/Users/dpfab/Documents/School/S2024/Neural Networks')

usecols = ["class", "cap-diameter", "cap-shape", "cap-color",
    "does-bruise-or-bleed", "gill-color", "stem-height", "stem-width",
    "stem-color", "has-ring", "habitat", "season"]
df = pd.read_csv('mushroom.csv', usecols=usecols)
print(df)

func_dicts = {}
dicts = {}

for x in df.to_dict().keys():
    #initialize new dictionary
    new_dict = {}
    for num, i in  enumerate(set(df[x])):
        #add conversion to number to new dict
        new_dict[i] = num
    df[x] = list(map(lambda s : new_dict[s], df[x]))
df.to_csv('mushroom_nums.csv', index=False)
