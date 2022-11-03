import json

with open("Dataset/result.json") as f:
    data = f.read()
js = json.loads(data)

print(js.get("1").get("Height"))