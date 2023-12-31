import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import geopandas as gpd
df = pd.read_csv('/home/project_docker/res_dpre.csv')

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# Create a bar plot
plt.figure(figsize=(16, 8))
sns.barplot(x='CouncilArea', y='Price_ln', data=df, estimator=len)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Count of Observations by Council Area')
plt.show()

plt.savefig('vis1.png')

# Calculate the top 5 most expensive homes based on 'Log_Price'
top_5_expensive_homes = df.nlargest(5, 'Price_ln')

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_5_expensive_homes['Suburb'], top_5_expensive_homes['Price_ln'], color='blue')
plt.xlabel('Suburb')
plt.ylabel('Price_ln')
plt.title('Top 5 Most Expensive Homes (Price_ln)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()

plt.savefig('vis2.png')

df.hist(figsize=(12, 10), bins=20)
plt.suptitle('Histograms of Numerical Features', y=0.95)
plt.show()

plt.savefig('vis3.png')

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longtitude'], df['Lattitude']))

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world.plot()

gdf.plot(ax=plt.gca(), marker='o', color='red', markersize=5)

plt.show()
plt.savefig('vis4.png')

numeric_columns = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount', 'Price_ln']

sns.pairplot(df[numeric_columns])
plt.show()

plt.savefig('vis5.png')
numeric_columns = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount', 'Price_ln']

sns.pairplot(df[numeric_columns])
plt.show()

plt.savefig('vis6.png')

plt.figure(figsize=(10, 6))
sns.countplot(x='Rooms', data=df, hue='Type', palette='viridis', dodge=True)
plt.title('Distribution of Rooms by Property Type')
plt.xlabel('Number of Rooms')
plt.ylabel('Count')
plt.show()
plt.savefig('vis7.png')

plt.figure(figsize=(10, 6))
sns.histplot(df['Distance'], bins=20, kde=True, color='green')
plt.title('Distribution of Distance to City')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()
plt.savefig('vis8.png')

plt.figure(figsize=(10, 6))
sns.histplot(df['Price_ln'], bins=30, kde=True, color='blue')
plt.title('Distribution of Property Prices')
plt.xlabel('Price_ln')
plt.ylabel('Frequency')
plt.show()

plt.savefig('vis9.png')
plt.figure(figsize=(12, 8))
sns.boxplot(x='Type', y='Price_ln', data=df)
plt.title('Boxplot of Property Type vs. Price')
plt.show()

plt.savefig('vis10.png')

plt.figure(figsize=(12, 6))
sns.countplot(x='Regionname', data=df, hue='Regionname', palette='Set1', dodge=True)

plt.xticks(rotation=45, ha='right')

plt.legend().set_visible(False)
plt.show()

plt.savefig('vis11.png')











