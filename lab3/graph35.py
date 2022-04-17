import pandas as pd
import plotly.express as px

# print(df.head)

df34 = pd.read_csv("lab3/graphData/34data.csv")

fig = px.line(
    df34,
    x="distortion",
    y="score",
    color="Pattern",
    labels={"scores": "Number of suceesfull recalls (out of 100)",},
    title="Number of successful recalls out of 100 per pattern and distortion level",
)
fig.show()
