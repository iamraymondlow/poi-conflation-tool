import json
import os
import model
import pandas as pd
import geopandas as gpd
from onemap_downloader import OneMap
from osm_processor import OSM
from sla_processor import SLA

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
            self.google_data = None

        # load formatted HERE Map data. If it does not exist, save as None.
        print('Loading HERE Map data from local directory...')
        if os.path.exists(os.path.join(os.path.dirname(__file__), config['here_cache'])):
            self.here_data = self._load_json_as_geopandas(config['here_cache'])
        else:
            self.here_data = None

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

    def extract_poi(self, lat, lng):
        """
        Extracts the neighbouring POIs based on the geograhical coordinates and performs POI conflation.
        """
        # extracts neighbouring POIs from OSM

        # extract neighbouring POIs from OneMap

        # extract neighbouring POIs from SLA

        # query neighbouring POIs from GoogleMap

        # query neighbouring POIs from HERE Map

        # perform conflation

        return None

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
