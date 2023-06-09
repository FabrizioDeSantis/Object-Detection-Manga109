import pandas as pd

file_path = "autori2.txt"
data = pd.read_csv(file_path, sep="\t", header=None)
data.columns = ["id", "author", "title"]
ids = data["id"].tolist()
authors = data["author"].tolist()
titles = data["title"].tolist()

count = 1

with open("autori22.txt", "w") as f:

    for a, t in zip(authors, titles):
        f.write(str(count) + "\t" + str(a) + "\t" + str(t) + "\n")
        count += 1
