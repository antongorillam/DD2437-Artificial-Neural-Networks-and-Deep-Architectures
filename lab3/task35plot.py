import pandas as pd
import plotly.express as px

# print(df.head)


df = pd.read_csv("graphData/3_5/0.00v2.csv")

#,Number of patterns,noise level,recall accuracy rate,perfect recall rate



df["Number of updates"] = [i for i in range(len(df["Number of patterns"]))]
fig = px.line(
    df,
    x="Number of patterns",
    y="perfect recall rate",
    title="Perfect recall rate per number of stored patterns"
)
fig.write_image("graphData/3_5/point2_graph_200_patterns.png")

