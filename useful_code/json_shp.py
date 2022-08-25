# shp to GeoJson
import geopandas as gpd
# data = gpd.read_file('中国省级区域20200720.shp')
# data.to_file("中国省级区域20200720.json", driver='GeoJSON', encoding="utf-8")

# GeoJson to shp
data = gpd.read_file(r'‪F:\he\data\building\test\grid_001.geojson')
data.to_file('中国省级区域20200720', driver='ESRI Shapefile', encoding='utf-8')





