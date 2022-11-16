import json

with open("Dataset/result.json") as f:
    data = f.read()
js = json.loads(data)

print(js.get("1").get("FreshWeightShoot"))
print(js.get("1").get("DryWeightShoot"))
print(js.get("1").get("Height"))
print(js.get("1").get("Diameter"))