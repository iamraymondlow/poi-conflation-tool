import pandas as pd
import os
import json
import re
import numpy as np
import geopandas as gpd
import pyproj
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy.fuzz import token_set_ratio
from shapely.geometry import Point
from functools import partial
from shapely.ops import transform


# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class Model:
    """
    This class perform model training for identifying duplicated POIs between different data sources.
    """
    def train_model(self):
        """
        This function performs model training for identifying duplicated POIs between different data
        sources.
        """
        # load manually labeled data
        manual_data = pd.read_csv(os.path.join(os.path.dirname(__file__), config['labeled_data']))
        manual_data = manual_data.head(n=50)  # TODO remember to remove
        manual_data['duplicates'] = manual_data['duplicates'].apply(self._format_duplicates)
        manual_data = manual_data[['properties.address.formatted_address', 'properties.name',
                                   'lat', 'lng', 'id', 'duplicates']]
        manual_data = gpd.GeoDataFrame(manual_data,
                                       geometry=gpd.points_from_xy(manual_data['lng'],
                                                                   manual_data['lat']))

        # process manually labeled data
        train_test_data = self._process_manual_data(manual_data)

        # train model based on processed labeled data

        # evaluate model performance on hold out set

        # save trained model

        return train_test_data


    def _format_duplicates(self, duplicate_string):
        """
        Extracts the IDs of the duplicated POIs in a list format.

        :param duplicate_string: str
            Contains the duplicated IDs in string format.
        :return:
        duplicates: list
            Contains the duplicated IDs in list format.
        """
        duplicates = re.sub('[\[\]\']', '', duplicate_string).split(', ')
        if len(duplicates) == 1 and duplicates[0] == '':
            return []
        else:
            return duplicates

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

    def _label_data(self, manual_data, centroid_idx, address_matrix):
        """
        Generates the labeled data for the neighbouring POIs around a centroid POI.

        :param manual_data: GeoDataFrame
            Contains the manually labeled data and the ID information of their duplicates.
        :param centroid_idx: int
            Contains the index of the centroid POI.
        :param address_matrix: np.array
            Contains the address matrix after vectorising the address corpus using TFIDF.
        :return:
        """
        # identify neighbouring POIs
        buffer = self._buffer_in_meters(manual_data.loc[centroid_idx, 'lng'],
                                        manual_data.loc[centroid_idx, 'lat'],
                                        config['search_radius'])
        neighbour_pois = manual_data[manual_data.intersects(buffer)]
        neighbour_idx = list(neighbour_pois.index)

        # calculate address similarity score for neighbouring POIs
        centroid_address = address_matrix[centroid_idx, :]
        address_similarity = cosine_similarity(address_matrix[neighbour_idx, :], centroid_address).reshape(-1, 1)

        # calculate name similarity score for neighbouring POIs
        if pd.isnull(manual_data.loc[centroid_idx, 'properties.name']):
            return None
        name_similarity = np.array([token_set_ratio(manual_data.loc[centroid_idx, 'properties.name'], neighbour_name)
                                    for neighbour_name in
                                    manual_data.loc[neighbour_idx, 'properties.name'].tolist()]).reshape(-1, 1)

        # extract labels for neighbouring POIs
        labels = np.zeros((len(neighbour_idx), 1))
        for i in range(len(neighbour_idx)):
            if manual_data.loc[neighbour_idx[i], 'id'] in manual_data.loc[centroid_idx, 'duplicates']:
                labels[i, 0] = 1
            else:
                pass

        return np.hstack((address_similarity, name_similarity, labels))


    def _process_manual_data(self, manual_data):
        """
        Processes the manually labeled data for model training and evaluation by identifying neighbouring POIs
        and labeling them as either duplicates or not duplicates.

        :param manual_data: GeoDataFrame
            Contains the manually labeled data and the ID information of their duplicates.
        :return:
        labeled_data: np.array
            Contains the labeled data ready for model training and evaluation.
        """
        address_corpus = manual_data['properties.address.formatted_address'].fillna('Singapore').tolist()
        address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        address_matrix = address_vectorizer.fit_transform(address_corpus)

        labeled_data = None
        for i in tqdm(range(len(manual_data))):
            temp_data = self._label_data(manual_data, i, address_matrix)

            if (temp_data is not None) and (labeled_data is not None):
                labeled_data = np.vstack((labeled_data, temp_data))
            elif temp_data is not None:
                labeled_data = temp_data
            else:
                pass

        return labeled_data


if __name__ == '__main__':
    model = Model()
    data = model.train_model()
