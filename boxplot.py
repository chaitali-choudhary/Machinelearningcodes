import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/Chaitali/Desktop/SIMLP 2019/Student.csv")
plt.boxplot(df["Total"])
plt.show()
