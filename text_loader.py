import os

addr = "data/"
dictionary = {}


for file in os.listdir(addr):
    with open(addr + file, "r", encoding="utf-8") as f:
        dictionary[file] = f.read().splitlines()

def get_data(what):
    try:
        return dictionary[what]
    except KeyError:
        return None