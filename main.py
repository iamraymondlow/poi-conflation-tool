import json
import os
import pandas as pd
import geopandas as gpd
import pyproj
import glob
import numpy as np
from googlemap_downloader import GoogleMapScrapper
from heremap_downloader import HereMapScrapper
from onemap_downloader import OneMap
from osm_processor import OSM
from sla_processor import SLA
from shapely.geometry import Point
from functools import partial
from shapely.ops import transform
from model import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy.fuzz import token_set_ratio
from joblib import load

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
        # load locally cached POIs that was conflated in the past
        print('Loading conflated data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['conflated_cache'])):
            if not os.path.exists(config['conflated_directory']):
                os.makedirs(config['conflated_directory'])
            self.conflated_data = None
        else:
            self.conflated_data = gpd.read_file(os.path.join(os.path.dirname(__file__), config['conflated_cache']))

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
        self.google_scrapper = GoogleMapScrapper(config['search_radius'])

        # load formatted HERE Map data. If it does not exist, save as None.
        print('Loading HERE Map data from local directory...')
        if os.path.exists(os.path.join(os.path.dirname(__file__), config['here_cache'])):
            self.here_data = self._load_json_as_geopandas(config['here_cache'])
        else:
            self.here_data = None
        self.here_scrapper = HereMapScrapper(config['search_radius'])

        # check if machine learning model is trained. If not, train model.
        model = Model()
        if not os.listdir(os.path.exists(os.path.join(os.path.dirname(__file__), config['models_directory']))):
            model.train_model()

        model_filenames = glob.glob(os.path.join(os.path.dirname(__file__),
                                                 config['models_directory'] + 'model_?.joblib'))
        self.models = [load(filename) for filename in model_filenames]

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
        conflated_pois: Geodataframe
            Contains the conflated POIs.
        """
        # create circular buffer around POI
        buffer = self._buffer_in_meters(lng, lat, config['search_radius'])

        # extracts neighbouring POIs from OSM
        osm_pois = self.osm_data[self.osm_data.intersects(buffer)]

        # extract neighbouring POIs from OneMap
        onemap_pois = self.onemap_data[self.onemap_data.intersects(buffer)]

        # extract neighbouring POIs from SLA
        sla_pois = self.sla_data[self.sla_data.intersects(buffer)]

        # extract neighbouring POIs from GoogleMap either locally or using API
        if (self.google_data is not None) and (stop_id in self.google_data['stop'].tolist()):
            google_pois = self.google_data[self.google_data['stop'] == stop_id]
        else:
            google_pois = self.google_scrapper.extract_poi(lat, lng, stop_id)
            if self.google_data is None:
                self.google_data = google_pois
            else:
                self.google_data = pd.concat([self.google_data, google_pois], ignore_index=True)

        # extract neighbouring POIs from HERE Map either locally or using API
        if (self.here_data is not None) and (stop_id in self.here_data['stop'].tolist()):
            here_pois = self.here_data[self.here_data['stop'] == stop_id]
        else:
            here_pois = self.here_scrapper.extract_poi(lat, lng, stop_id)
            if self.here_data is None:
                self.here_data = here_pois
            else:
                self.here_data = pd.concat([self.here_data, here_pois], ignore_index=True)

        # perform conflation
        combined_pois = pd.concat([osm_pois, onemap_pois, sla_pois,
                                   google_pois, here_pois], ignore_index=True)
        conflated_pois = self._perform_conflation(combined_pois)

        # cache conflated POIs
        self.conflated_data = pd.concat([self.conflated_data, conflated_pois], ignore_index=True)
        self.conflated_data.to_file(os.path.join(os.path.dirname(__file__), config['conflated_cache']))

        return conflated_pois

    def _perform_conflation(self, potential_duplicates):
        """
        Performs POIs conflation by identifying the duplicates among a group of POIs.

        :param potential_duplicates: Geodataframe
            Contains the group of POIs found in a particular region. Some of which could be duplicates.

        :return:
        conflated_pois: Geodataframe
            Contains the conflated POIs.
        """
        # remove POIs with no name information
        potential_duplicates.dropna(subset=['properties.name'], inplace=True)
        potential_duplicates = potential_duplicates[~potential_duplicates['properties.name'].isin(['None'])]
        potential_duplicates.reset_index(drop=True, inplace=True)

        # vectorise address information
        address_corpus = potential_duplicates['properties.address.formatted_address'].fillna('Singapore').tolist()
        address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        address_matrix = address_vectorizer.fit_transform(address_corpus)

        # identify duplicates using ML models
        duplicate_idx = []
        for i in range(len(potential_duplicates)):
            duplicate_idx.append(self._identify_duplicates(potential_duplicates, i, address_matrix))

        assert len(potential_duplicates) == len(duplicate_idx)
        potential_duplicates['duplicates'] = duplicate_idx

        # conflate duplicated POIs
        conflated_pois = None

        return conflated_pois

    def _identify_duplicates(self, data, centroid_idx, address_matrix):
        """
        Identifies duplicated POI points to a centroid POI by passing the name similarity and address
        similarity values into machine learning models for classification.

        :param data: Geodataframe
            Contains information about the neighbouring POIs and the centroid POI.
        :param centroid_idx: int
            Contains the index of the centroid POI in data.
        :param address_matrix: np.array
            Contains the TFIDF-vectorised address matrix.

        :return:
        duplicated_id: list
            Contains a list of IDs for the duplicated POIs
        """
        if len(data) > 1:  # presence of neighbouring POIs
            data.reset_index(drop=True, inplace=True)
            neighbour_idx = list(data.index)
            neighbour_idx.remove(centroid_idx)
            neighbour_data = data.iloc[neighbour_idx]
            print(centroid_idx)
            print(neighbour_idx)
            print(data.index)

            address_similarity = cosine_similarity(address_matrix[neighbour_idx, :],
                                                   address_matrix[centroid_idx, :]).reshape(-1, 1)

            name_similarity = np.array([token_set_ratio(data.loc[centroid_idx, 'properties.name'], neighbour_name)
                                        for neighbour_name
                                        in neighbour_data['properties.name'].tolist()]).reshape(-1, 1)

            # Pass name and address similarity values into ML models
            predict_prob = np.zeros((len(neighbour_idx), 2))
            for model in self.models:
                predict_prob += model.predict_proba(np.hstack((address_similarity, name_similarity)))
            temp_idx = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
            print(predict_prob)
            print(temp_idx)
            # POIs are only considered as duplicates if they come from different sources
            duplicate_id = [data.loc[neighbour_idx[idx], 'id'] for idx in temp_idx
                            if data.loc[neighbour_idx[idx], 'properties.source'] !=
                            data.loc[centroid_idx, 'properties.source']]

            return duplicate_id
        else:  # no neighbours
            return []


if __name__ == '__main__':
    tool = POIConflationTool()
    osm_data = tool.osm_data
    onemap_data = tool.onemap_data
    sla_data = tool.sla_data
    google_data = tool.google_data
    heremap_data = tool.here_data
    models = tool.models
    # data = tool.extract_poi(1.3414, 103.9633, 'test')

    # data = pd.read_excel('data/hvp_data/combined_stop_data.xlsx')
    # for i in range(len(data)):
    #     print('Processing {}/{}'.format(i+1, len(data)))
    #     tool.extract_poi(data.loc[i, 'StopLat'],
    #                      data.loc[i, 'StopLon'],
    #                      str(data.loc[i, 'StopID']))


