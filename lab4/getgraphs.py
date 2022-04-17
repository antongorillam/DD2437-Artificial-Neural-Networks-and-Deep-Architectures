import pandas as pd
import plotly.express as px

df = px.data.gapminder().query("continent=='Oceania'")
# print(df.head)


df200 = pd.read_csv("lab4/isac/recon_losses_nHidden_200_n_iterations_200.csv")
df300 = pd.read_csv("lab4/isac/recon_losses_nHidden_300_n_iterations_200.csv")
df400 = pd.read_csv("lab4/isac/recon_losses_nHidden_400_n_iterations_200.csv")
df500 = pd.read_csv("lab4/isac/recon_losses_nHidden_500_n_iterations_200.csv")


df200["recon_losses"] = df200["recon_losses"] / 60000
df300["recon_losses"] = df300["recon_losses"] / 60000
df400["recon_losses"] = df400["recon_losses"] / 60000
df500["recon_losses"] = df500["recon_losses"] / 60000

df200["nodes"] = [200 for i in range(21)]
df300["nodes"] = [300 for i in range(21)]
df400["nodes"] = [400 for i in range(21)]
df500["nodes"] = [500 for i in range(21)]
frames = [df200, df300, df400, df500]
mergedDf = pd.concat(frames)
print(mergedDf)
# print(df200.head)

fig = px.line(
    mergedDf,
    x="num_iterations",
    y="recon_losses",
    color="nodes",
    labels={
        "recon_losses": "Average reconstruction loss",
        "num_iterations": "Number of epochs",
        "nodes": "Number of hidden units",
    },
    title="RBM average reconstruction loss per epoch by number of hidden units",
)
fig.show()
