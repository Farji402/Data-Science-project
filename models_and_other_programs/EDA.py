import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Reading the data_frame
df = pd.read_csv('train.csv')

"""#shape of the data_frame
print('\n', df.info(),'\n')

#changing null values in price with their mean.
df['price'].replace(np.nan, df['price'].mean(), inplace= True)

print('\n')
#printing unique values in each column
for i in df.columns:
    print( i,": ", len(df[i].unique()))

print('\n')


#Checking relation between rating and country of wine
df_country = df.groupby('country')['points'].mean()

sns.set()
df_country.plot(kind= 'bar', style= 'step')
plt.ylabel('mean points')
plt.xticks(rotation= 90)
plt.tight_layout()
plt.show()

#Checking relation between price and points
sns.lmplot(x= 'price', y= 'points', data= df)
plt.tight_layout()
plt.show()



#Checking the extremes of the points
sns.set_palette("cubehelix")
plt.hist(df['points'], density= True)
plt.xlabel('Points given by users')
plt.ylabel("Percentage distribution")
plt.show()

#price distrution grouped by points received
data_lim = df[df['price'] < 2500]
sns.boxplot(x= 'points', y= 'price', data= data_lim, whis= 20)
plt.show()


#countries with the variety they have produced the most
df_new  =  pd.crosstab(df['country'].astype('category'), df['variety'].astype('category'))

for i in df_new.index:

    df_new1 = df_new.loc[i].nlargest(1)

    print(i,": ",df_new1.index[0])
"""

#price distribution grouped by variety
df_price_lim = df[df['price'] < 1500]
sns.set_style('darkgrid')
g = sns.pointplot(x= 'variety', y= 'price', data= df, palette= 'husl')
plt.ylabel('Mean Price')
plt.xticks(rotation= 90)
plt.tight_layout()
plt.show()
