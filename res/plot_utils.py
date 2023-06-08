import json
import pandas as pd
import matplotlib.pyplot as plt

with open("results.txt") as f:
  data = f.read()
elements = []
start_index = 0
end_index = 0

for i in range(len(data)):
    if data[i] == '{':
        start_index = i
    elif data[i] == '}':
        end_index = i
        elements.append(data[start_index:end_index+1])

js = []
for element in elements:
  js.append(json.loads(element))

print(js)

for i, d in enumerate(js):
  kw = next(iter(d))
  data = pd.DataFrame.from_dict(d)
  fig = data.plot(kind="bar", xlabel=str(kw), x=str(kw), rot=0, figsize=(10, 10)).get_figure()
  fig.savefig("results-" + str(kw) + ".png")