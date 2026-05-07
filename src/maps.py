import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
from cmcrameri import cm as cmc
import numpy as np

# -------------------------------------------------
# 4.1.3. Airbnb/ Map of Zurich with airbnb listings for EDA
# -------------------------------------------------

# helper function to calculate color from price
def get_color(price):
    if price < 100:
        return "green"
    elif price < 200:
        return "blue"
    elif price < 300:
        return "orange"
    else:
        return "red"
    
# main function for creating the map
def zurich_map_eda(airbnb_gdf: gpd, forest: gpd, quartiere: gpd) -> map:
    # initiate folium map canvas
    m = folium.Map(
        location=[47.3769, 8.5417],
        zoom_start=12,
        min_zoom=12,
        max_zoom=17,
        tiles="CartoDB dark-matter" # the bright map would be CartoDB positron
        )
    
    # load forest
    folium.GeoJson(forest,
               name="Forested areas",
               style_function=lambda x: {
                   "color": "green",
                   "weight": 1,
                   "opacity": 0.5,
                   "fillOpacity": 0.2
                   }
              ).add_to(m)
    
    # load quartiere borders
    folium.GeoJson(quartiere,
               name="Quartier borders",
               style_function=lambda x: {
                   "color": "white",
                   "weight": 1,
                   "opacity": 0.7,
                   "fillOpacity": 0
                   },
               tooltip=folium.GeoJsonTooltip(
                   fields=["qname"],
                   aliases=["Quartier:"]
                   )
              ).add_to(m)
    
    # create colormap for listings price
    log_price = np.log1p(airbnb_gdf["price"])

    colormap = cm.LinearColormap(
        list(reversed(cm.linear.YlGnBu_09.colors)),
        vmin=log_price.min(),
        vmax=log_price.max()
    )
    
    # finally, add airbnb listings
    listings_layer = folium.FeatureGroup(name="Airbnb Listings")
    for _, row in airbnb_gdf.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            stroke=False,
            fill=True,
            fill_color=colormap(np.log1p(row["price"])),
            fill_opacity=1,
            tooltip=f"{row['price']} CHF"
        ).add_to(listings_layer)
    listings_layer.add_to(m)

    # add the legend for the pricing colormap
    legend_html = """
    <div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    z-index: 9999;
    background-color: rgba(0,0,0,0.7);
    color: white;
    padding: 10px;
    border-radius: 6px;
    font-size: 14px;
    ">
    <b>Price (CHF)</b><br>
    <i style="background:#225ea8;width:10px;height:10px;display:inline-block"></i> Very low<br>
    <i style="background:#1d91c0;width:10px;height:10px;display:inline-block"></i> low<br>
    <i style="background:#41b6c4;width:10px;height:10px;display:inline-block"></i> Medium<br>
    <i style="background:#a1dab4;width:10px;height:10px;display:inline-block"></i> High<br>
    <i style="background:#ffffcc;width:10px;height:10px;display:inline-block"></i> Very high<br>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    # add toggle for layers and return map
    folium.LayerControl().add_to(m)
    return m


# -------------------------------------------------
# 4.2.2. Housing stock/ Geographic distribution of airbnb stock
# 4.3.1. Rental price/ Geographic distribution of rental prices
# -------------------------------------------------
def quartier_map(gdf: gpd.GeoDataFrame, column: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    gdf.plot(ax=ax,
             column=column,
             cmap=cmc.lapaz,
             legend=True)
    ax.axis("off")
    plt.title(title)
    plt.tight_layout

