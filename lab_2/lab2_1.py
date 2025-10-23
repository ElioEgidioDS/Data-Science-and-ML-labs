import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib.pyplot as plt


from lab2_1_geogrid import GridDiscretizer

img_path = r"C:\Users\UTENTE\Desktop\data science\Data-Science-and-ML-labs\lab_2\NYC_POIs\New_York_City_Map.png"
img = mpimg.imread(img_path)
ny_poi = pd.read_csv(r'C:\Users\UTENTE\Desktop\data science\Data-Science-and-ML-labs\lab_2\NYC_POIs\pois_all_info',sep='\t')
ny_municipality = pd.read_csv(r'C:\Users\UTENTE\Desktop\data science\Data-Science-and-ML-labs\lab_2\NYC_POIs\ny_municipality_pois_id.csv',sep=',',header=None,names=['@id'])


def plotOnImage(category, categories, view):

    category_top = categories[category]

    data = view[view[category].notna()]

    working_data = data[data[category].isin(category_top.index)]

    fig, ax = plt.subplots(figsize=(12,12))

    lat_min, lat_max = 40.5024225, 40.9139069
    lon_min, lon_max = data['@lon'].min(), data['@lon'].max()
    
    # Show map image
    ax.imshow(img, extent=[lon_min, lon_max,
                           lat_min, lat_max],
              aspect='auto')
    ax.scatter(working_data['@lon'], working_data['@lat'], s=10)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"POIs for category: {category}")
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside
    plt.show()

    #WRONG: nonemptycat = view[category].notna
    #list = [nonemptycat[['@lat'],['@lon']]]

    #CORRECT: data = view[view[category].notna()]
    #coords = data[['@lat', '@lon']]



print(ny_poi)
print()
print(ny_municipality)

view = ny_poi[ny_poi['@id'].isin(ny_municipality['@id'])]
print(view) 

missing_values = view.isna().sum()
print(missing_values)

print()
#print(view.groupby(['amenity']).size()) VIABLE OPTION

#PREFERRED OPTION TO THE PLO
amenity_counts = ny_poi['amenity'].value_counts()
shop_counts = ny_poi['shop'].value_counts()
public_transport_counts = ny_poi['public_transport'].value_counts()
highway_counts = ny_poi['highway'].value_counts()

amenity_top = amenity_counts.head(10)
shop_top = shop_counts.head(10)
public_top = public_transport_counts.head(10)
highway_top = highway_counts.head(10)

print(amenity_counts)
print(shop_counts)
print(public_transport_counts)
print(highway_counts)

fig, axes = plt.subplots(2,2,figsize=(15,10))
axes = axes.flatten()

categories = {
    "amenity": amenity_top,
    "shop": shop_top,
    "public_transport": public_top,
    "highway": highway_top
}

'''for ax, (title, series) in zip(axes, categories.items()):
    series.plot(kind='bar', ax=ax)
    ax.set_title(f"Top {title} Types")
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

plotOnImage("amenity",categories, view)
plt.tight_layout()
plt.show()
 '''

#2.5: discretize data and show it in the map

grid = GridDiscretizer(img, view, categories)
discretized = grid.discretize()
print(discretized.head())

grid.plotGrid("amenity")

#2.6 new dataframe for study on type frequency per grid per category

'''MY FIRST TRY AT A SOLUTION:
    filtered= view[view["amenity"].notna()]
    types = filtered["amenity"].unique()

    dataf = pd.DataFrame(view,axes=discretized['cell_id'], columns=types)
    dataf.groupby(["cell_id"]).count()

'''

'''ACTUAL SOLUTION:'''
counts_amenity = (
    discretized
    .groupby(["cell_id", "amenity"])
    .size()
    .unstack(fill_value=0)
)

print(counts_amenity)

'''OTHER POSSIBLE SOLUTIONS:

-> counts = pd.crosstab(discretized['cell_id'], discretized['amenity'])

-> counts = discretized.pivot_table(index='cell_id', columns='amenity', values='@id', aggfunc='count', fill_value=0)
'''
counts_shop = (
    discretized
    .groupby(["cell_id", "shop"])
    .size()
    .unstack(fill_value=0)
)

#2.7 correlation btw places and categories

# 1. Compute the cell-type counts
counts_amenity = discretized.groupby(['cell_id','amenity']).size().unstack(fill_value=0)
counts_shop = discretized.groupby(['cell_id','shop']).size().unstack(fill_value=0)

# 2. Compute correlation matrices
corr_amenity = counts_amenity.corr()
corr_shop = counts_shop.corr()

# 3. Plot
plt.figure(figsize=(10,8))
sns.heatmap(corr_amenity, cmap='coolwarm', annot=False)
plt.title("Amenity POI Type Correlation (Spatial Co-occurrence)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(corr_shop, cmap='coolwarm', annot=False)
plt.title("Shop POI Type Correlation (Spatial Co-occurrence)")
plt.show()

# 4. See top correlated pairs
top_pairs = corr_amenity.unstack().sort_values(ascending=False)
top_pairs = top_pairs[top_pairs < 0.9999]  # remove self-correlations
print(top_pairs.head(10))


