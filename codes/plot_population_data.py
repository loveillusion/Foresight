import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('out.csv')

df.plot(x="Population", y="Number of Beds")
plt.show()
