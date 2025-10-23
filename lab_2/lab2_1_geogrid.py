import numpy as np
import matplotlib.pyplot as plt

class GridDiscretizer:

    def __init__(self, image, poi_view, top_categories):

        self.view = poi_view
        self.categories = top_categories
        self.image =  image

        self.lat_min = 40.5024225
        self.lat_max = 40.9139069
        self.lon_min = self.view['@lon'].min()
        self.lon_max = self.view['@lon'].max()

        self.gridY = np.linspace(self.lat_min,self.lat_max,21) 
        self.gridX = np.linspace(self.lon_min,self.lon_max,21)     

    def discretize(self):
        data = self.view.copy()

        x_bin = np.digitize(data['@lon'], self.gridX) - 1
        y_bin = np.digitize(data['@lat'], self.gridY) - 1


        #we turn indices into a single cell id
        n_cols = len(self.gridX) - 1
        data['cell_id'] = y_bin * n_cols + x_bin


        self.discretized_data = data
        return data
    
    def plotGrid(self, category=None):

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.image, extent=[self.lon_min, self.lon_max, self.lat_min, self.lat_max], aspect='auto')

        for x in self.gridX:
            ax.axvline(x, color='black', linewidth = 0.5)
        for y in self.gridY:
            ax.axhline(y, color='black', linewidth=0.5)

        if category and category in self.categories:
            subset = self.discretized_data[self.discretized_data[category].notna()]
            unique_types = subset[category].unique()
            cmap = plt.get_cmap('tab10')  # or 'tab20', 'Set3', etc.

            for i, t in enumerate(unique_types):
                color = cmap(i % cmap.N)
                subset_t = subset[subset[category] == t]
                ax.scatter(subset_t['@lon'], subset_t['@lat'],
                            s=10, alpha=0.6, label=t, color=color)

            ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title("nyc pois distribution discretized")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

        plt.show()


