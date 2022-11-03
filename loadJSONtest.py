import json

with open("Dataset/result.json") as f:
    data = f.read()
js = json.loads(data)

print(len(js))