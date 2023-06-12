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

for i, d in enumerate(js):
  kw = next(iter(d))
  data = pd.DataFrame.from_dict(d)
  fig, ax = plt.subplots(figsize=(15, 10))
  if kw=="classes":
     data.plot(kind="bar", x=str(kw), y=["mAP", "mAP (single)"], rot=0, ax=ax)
  else:
    data.plot(kind="bar", x=str(kw), y=["mAP", "mAP_50", "mAP_75"], rot=0, ax=ax)
  # Add values on top of bars
  for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

  plt.xlabel(str(kw))
  plt.ylabel("Value")
  plt.title("Results")
  plt.legend()
  #plt.show()
  fig.savefig("results-" + str(kw) + ".png")