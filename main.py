import json
import os
import train_model
import pandas as pd
import geopandas as gpd
import pyproj
from onemap_downloader import OneMap
from osm_processor import OSM
from sla_processor import SLA
from shapely.geometry import Point
from functools import partial
from shapely.ops import transform

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class POIConflationTool:
    """
    Performs POI conflation based on 5 different data sources (i.e., OSM, OneMap, SLA, Google Maps
    and HERE Map).
    """
    def __init__(self):
        """
        Checks if the OSM, OneMap, and SLA datasets are formatted. If any of the datasets are
        not formatted, the appropriate functions will be triggered to begin formatting the dataset.
        Also checks if machine learning model for identifying POI duplicates are trained.
        """
        # load formatted OneMap data. If it does not exist, format and load data.
        print('Loading OneMap data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['onemap_cache'])):
            OneMap().format_data()
        self.onemap_data = self._load_json_as_geopandas(config['onemap_cache'])

        # load formatted SLA data. If it does not exist, format and load data.
        print('Loading SLA data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['sla_cache'])):
            SLA().format_data()
        self.sla_data = self._load_json_as_geopandas(config['sla_cache'])

        # load formatted OSM data. If it does not exist, format and load data.
        print('Loading OSM data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['osm_cache'])):
            OSM().format_data()
        self.osm_data = self._load_json_as_geopandas(config['osm_cache'])

        # load formatted Google data. If it does not exist, save as None.
        print('Loading Google data from local directory...')
        if os.path.exists(os.path.join(os.path.dirname(__file__), config['google_cache'])):
            self.google_data = self._load_json_as_geopandas(config['google_cache'])
        else:
            self.google_data = gpd.GeoDataFrame()

        # load formatted HERE Map data. If it does not exist, save as None.
        print('Loading HERE Map data from local directory...')
        if os.path.exists(os.path.join(os.path.dirname(__file__), config['here_cache'])):
            self.here_data = self._load_json_as_geopandas(config['here_cache'])
        else:
            self.here_data = gpd.GeoDataFrame()

        # check if machine learning model is trained. If not, train model.
        # if not os.listdir(os.path.exists(os.path.join(os.path.dirname(__file__), config['models_directory']))):
        #     model.train_model()
        self.models = None

    def _load_json_as_geopandas(self, cache_directory):
        """
        This function loads a local JSON file and converts it into a Geodataframe.

        :param cache_directory: str
            Contains the directory for the cache POI data file.
        :return:
        data: geopandas
            Contains the cached POI data file formatted as a Geodataframe.
        """
        with open(os.path.join(os.path.dirname(__file__), cache_directory)) as file:
            data_json = json.load(file)
        data = pd.json_normalize(data_json['features'])
        data = gpd.GeoDataFrame(data,
                                geometry=gpd.points_from_xy(data['geometry.lng'],
                                                            data['geometry.lat']))
        return data

    def _buffer_in_meters(self, lng, lat, radius):
        """
        Converts a latitude, longitude coordinate pair into a buffer with user-defined radius.s

        :param lng: float
            Contains the longitude information.
        :param lat: float
            Contains the latitude information.
        :param radius: float
            Contains the buffer radius in metres.
        :return:
        buffer_latlng: Polygon
            Contains the buffer.
        """
        proj_meters = pyproj.Proj(init='epsg:3414')  # EPSG for Singapore
        proj_latlng = pyproj.Proj(init='epsg:4326')

        project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
        project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)

        pt_meters = transform(project_to_meters, Point(lng, lat))

        buffer_meters = pt_meters.buffer(radius)
        buffer_latlng = transform(project_to_latlng, buffer_meters)
        return buffer_latlng

    def extract_poi(self, lat, lng, stop_id=None):
        """
        Extracts the neighbouring POIs around a location based on the geographical coordinates
        and performs POI conflation.

        :param lat: float
            Contains the latitudinal information.
        :param lng: float
            Contains the longitudinal information.
        :param stop_id: str
            Contains the unique ID of a particular stop.
        :return:
        """
        # create circular buffer around POI
        buffer = self._buffer_in_meters(lng, lat, config['search_radius'])

        # extracts neighbouring POIs from OSM
        osm_intersect = self.osm_data[self.osm_data.intersects(buffer)]

        # extract neighbouring POIs from OneMap
        onemap_intersect = self.onemap_data[self.onemap_data.intersects(buffer)]

        # extract neighbouring POIs from SLA
        sla_intersect = self.sla_data[self.sla_data.intersects(buffer)]

        # query neighbouring POIs from GoogleMap
        # google_intersect = self.google_data[self.google_data.intersects(buffer)]

        # query neighbouring POIs from HERE Map
        # here_intersect = self.here_data[self.here_data.intersects(buffer)]

        # perform conflation
        combined_intersect = pd.concat([osm_intersect, onemap_intersect,
                                        sla_intersect], ignore_index=True)

        return combined_intersect

    def _perform_conflation(self):
        return None


if __name__ == '__main__':
    tool = POIConflationTool()
    osm_data = tool.osm_data
    onemap_data = tool.onemap_data
    sla_data = tool.sla_data
    google_data = tool.google_data
    heremap_data = tool.here_data
    models = tool.models
    data = tool.extract_poi(1.3414, 103.9633)
