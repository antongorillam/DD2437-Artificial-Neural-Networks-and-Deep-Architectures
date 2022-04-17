import pandas as pd
import plotly.express as px

# print(df.head)


#dfp1 = pd.read_csv("graphData/3_3/p1.csv")
#dfp2 = pd.read_csv("graphData/3_3/p2.csv")
#dfp3 = pd.read_csv("graphData/3_3/p3.csv")
dfp4 = pd.read_csv("graphData/3_3/p4.csv")
dfp5 = pd.read_csv("graphData/3_3/p5.csv")
dfp6 = pd.read_csv("graphData/3_3/p6.csv")
dfp7 = pd.read_csv("graphData/3_3/p7.csv")
dfp8 = pd.read_csv("graphData/3_3/p8.csv")
dfp9 = pd.read_csv("graphData/3_3/p9.csv")
dfp10 = pd.read_csv("graphData/3_3/p10.csv")
dfp11 = pd.read_csv("graphData/3_3/p11.csv")

dfpoint4 = pd.read_csv("graphData/3_3/point4.csv")
dfpoint5 = pd.read_csv("graphData/3_3/point5.csv")


dfs = [
    #dfp1,
    #dfp2,
    #dfp3,
    dfp4,
    dfp5,
    dfp6,
    dfp7,
    dfp8,
    dfp9,
    dfp10,
    dfp11,
    dfpoint4,
    dfpoint5,
]
names = [
    #"pattern_1",
    #"pattern_2",
    #"pattern_3",
    "pattern_4",
    "pattern_5",
    "pattern_6",
    "pattern_7",
    "pattern_8",
    "pattern_9",
    "pattern_10",
    "pattern_11",
    "point_4",
    "point_5",
]

for i, df in enumerate(dfs):
    df["Number of updates"] = [i for i in range(len(df["num_iterations"]))]
    fig = px.line(
        df,
        x="Number of updates",
        y="energy",
        labels={"energy": "Energy",},
        title="Energy per number of flips, " + names[i],
    )
    fig.write_image("graphData/3_3/" + names[i] + "_graph.png")

