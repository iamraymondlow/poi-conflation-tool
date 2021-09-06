import json
import os
import pandas as pd
import geopandas as gpd
import pyproj
import glob
import numpy as np
from googlemap import GoogleMapScrapper
from heremap import HereMapScrapper
from onemap import OneMap
from osm import OSM
from sla import SLA
from shapely.geometry import Point
from shapely.ops import transform
from model import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy.fuzz import token_set_ratio, ratio
from joblib import load
from shapely import geometry
from util import divide_bounding_box, pixelise_region
from copy import deepcopy

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class POIConflationTool:
    """
    Performs POI conflation based on 5 different data sources (i.e., OSM, OneMap, SLA, Google Maps
    and HERE Map).
    """
    def __init__(self, subzones=None):
        """
        Checks if the OSM, OneMap, and SLA datasets are formatted. If any of the datasets are
        not formatted, the appropriate functions will be triggered to begin formatting the dataset.
        Also checks if machine learning model for identifying POI duplicates are trained.
        """
        # load country shapefiles for filtering out POIs based on subzones
        country_shp = gpd.read_file(os.path.join(os.path.dirname(__file__), config['country_shapefile']),
                                    encoding='utf-8')
        country_shp = country_shp.to_crs("EPSG:4326")
        if subzones is not None:
            country_shp = country_shp[country_shp['PLN_AREA_N'].isin(subzones)].reset_index(drop=True)
        self.country_shp = country_shp

        # load formatted OneMap data. If it does not exist, format and load data.
        print('Loading OneMap data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['onemap_cache'])):
            OneMap().format_data()
        onemap_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__), config['onemap_cache']))
        if subzones is not None:
            for i in range(len(country_shp)):
                if i == 0:
                    self.onemap_data = onemap_data[onemap_data.intersects(
                        country_shp.loc[i, 'geometry'])].reset_index(drop=True)
                else:
                    self.onemap_data = pd.concat([self.onemap_data,
                                                  onemap_data[onemap_data.intersects(
                                                      country_shp.loc[i, 'geometry'])].reset_index(drop=True)
                                                  ], ignore_index=True)
        else:
            self.onemap_data = onemap_data

        # load formatted SLA data. If it does not exist, format and load data.
        print('Loading SLA data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['sla_cache'])):
            SLA().format_data()
        sla_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__), config['sla_cache']))
        if subzones is not None:
            for i in range(len(country_shp)):
                if i == 0:
                    self.sla_data = sla_data[sla_data.intersects(
                        country_shp.loc[i, 'geometry'])].reset_index(drop=True)
                else:
                    self.sla_data = pd.concat([self.sla_data,
                                               sla_data[sla_data.intersects(
                                                   country_shp.loc[i, 'geometry'])].reset_index(drop=True)
                                               ], ignore_index=True)
        else:
            self.sla_data = sla_data

        # load formatted OSM data. If it does not exist, format and load data.
        print('Loading OSM data from local directory...')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['osm_cache'])):
            OSM().format_data()
        osm_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__), config['osm_cache']))
        if subzones is not None:
            for i in range(len(country_shp)):
                if i == 0:
                    self.osm_data = osm_data[osm_data.intersects(
                        country_shp.loc[i, 'geometry'])].reset_index(drop=True)
                else:
                    self.osm_data = pd.concat([self.osm_data,
                                               osm_data[osm_data.intersects(
                                                   country_shp.loc[i, 'geometry'])].reset_index(drop=True)
                                               ], ignore_index=True)
        else:
            self.osm_data = osm_data

        # load formatted Google data. If it does not exist, save as None.
        print('Loading Google data from local directory...')
        self.google_scrapper = GoogleMapScrapper(config['search_radius'])
        if subzones is None:
            if os.path.exists(os.path.join(os.path.dirname(__file__), config['google_cache'])):
                self.google_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__),
                                                                             config['google_cache']))
            else:
                self.google_data = None
        else:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), config['google_area_cache'])):
                self.google_data = self.google_scrapper.extract_area(subzones=subzones)
            google_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__),
                                                                    config['google_area_cache']))
            self.google_data = google_data[google_data.intersects(
                country_shp.loc[0, 'geometry'])].reset_index(drop=True)

        # load formatted HERE Map data. If it does not exist, save as None.
        print('Loading HERE Map data from local directory...')
        self.here_scrapper = HereMapScrapper(config['search_radius'])
        if subzones is None:
            if os.path.exists(os.path.join(os.path.dirname(__file__), config['here_cache'])):
                self.here_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__),
                                                                           config['here_cache']))
            else:
                self.here_data = None
        else:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), config['here_area_cache'])):
                self.here_data = self.here_scrapper.extract_area(subzones=subzones)
            here_data = self._load_json_as_geopandas(os.path.join(os.path.dirname(__file__),
                                                                  config['here_area_cache']))
            self.here_data = here_data[here_data.intersects(country_shp.loc[0, 'geometry'])].reset_index(drop=True)

        # check if machine learning model is trained. If not, train model.
        model = Model()
        if not os.listdir(os.path.join(os.path.dirname(__file__), config['models_directory'])):
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
        proj_meters = pyproj.CRS('EPSG:3414')  # EPSG for Singapore
        proj_latlng = pyproj.CRS('EPSG:4326')

        project_to_metres = pyproj.Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform
        project_to_latlng = pyproj.Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform
        pt_meters = transform(project_to_metres, Point(lng, lat))
        buffer_meters = pt_meters.buffer(radius)
        buffer_latlng = transform(project_to_latlng, buffer_meters)
        return buffer_latlng

    def conflate_area_poi(self):
        """
        Performs POI conflation within a certain user defined area and saves the conflated data locally.
        """
        # pixelise region based on shapefile
        coordinate_list = divide_bounding_box(max_lat=config['max_lat'], min_lat=config['min_lat'],
                                              max_lng=config['max_lng'], min_lng=config['min_lng'],
                                              querybox_dim=config['search_radius']*2)
        coordinate_list = pixelise_region(coordinate_list, self.country_shp)

        # extract POI in the region and perform conflation
        i = 1
        for coordinate in coordinate_list:
            print('Processing query {}/{}'.format(i, len(coordinate_list)))
            i += 1
            centre_lat = (coordinate[2] + coordinate[0]) / 2
            centre_lng = (coordinate[1] + coordinate[3]) / 2

            # create circular buffer around centre
            buffer = self._buffer_in_meters(centre_lng, centre_lat, config['search_radius'])

            # extracts neighbouring POIs from OSM
            osm_pois = self.osm_data[self.osm_data.intersects(buffer)]

            # extract neighbouring POIs from OneMap
            onemap_pois = self.onemap_data[self.onemap_data.intersects(buffer)]

            # extract neighbouring POIs from SLA
            sla_pois = self.sla_data[self.sla_data.intersects(buffer)]

            # extract neighbouring POIs from GoogleMap either locally or using API
            assert self.google_data is not None
            google_pois = self.google_data[self.google_data.intersects(buffer)]

            # extract neighbouring POIs from HERE Map either locally or using API
            assert self.here_data is not None
            here_pois = self.here_data[self.here_data.intersects(buffer)]

            # perform conflation
            combined_pois = pd.concat([osm_pois, onemap_pois, sla_pois,
                                       google_pois, here_pois], ignore_index=True)
            conflated_pois = self._perform_conflation(combined_pois)

            # cache conflated POIs
            if (self.conflated_data is None) and (conflated_pois is None):
                continue
            elif self.conflated_data is None:
                self.conflated_data = conflated_pois
            else:
                self.conflated_data = pd.concat([self.conflated_data, conflated_pois], ignore_index=True)

            # if self.conflated_data is not None:
            #     if 'duplicates' in self.conflated_data.columns:
            #         self.conflated_data.drop(columns=['duplicates'], inplace=True)
            #     self.conflated_data.to_file(os.path.join(os.path.dirname(__file__), config['conflated_area_cache']),
            #                                 encoding='utf-8')

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
        buffer = self._buffer_in_meters(float(lng), float(lat), config['search_radius'])

        # extracts neighbouring POIs from OSM
        osm_pois = self.osm_data[self.osm_data.intersects(buffer)]

        # extract neighbouring POIs from OneMap
        onemap_pois = self.onemap_data[self.onemap_data.intersects(buffer)]

        # extract neighbouring POIs from SLA
        sla_pois = self.sla_data[self.sla_data.intersects(buffer)]

        # extract neighbouring POIs from GoogleMap either locally or using API  #TODO needs to be updated
        # if (self.google_data is not None) and (stop_id in self.google_data['stop'].tolist()):
        #     google_pois = self.google_data[self.google_data['stop'] == stop_id]
        # else:
        #     google_pois = self.google_scrapper.extract_poi(lat, lng, stop_id)
        #     if self.google_data is None:
        #         self.google_data = google_pois
        #     else:
        #         self.google_data = pd.concat([self.google_data, google_pois], ignore_index=True)

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
                                   here_pois], ignore_index=True)  #TODO remove
        # combined_pois = pd.concat([osm_pois, onemap_pois, sla_pois,
        #                            google_pois, here_pois], ignore_index=True)

        if len(combined_pois) == 0:
            return None

        conflated_pois = self._perform_conflation(combined_pois)

        # cache conflated POIs
        if (conflated_pois is None) and (self.conflated_data is None):
            pass
        elif self.conflated_data is None:
            self.conflated_data = conflated_pois
        else:
            self.conflated_data = pd.concat([self.conflated_data, conflated_pois], ignore_index=True)

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
        if len(potential_duplicates) > 1:
            address_corpus = potential_duplicates['properties.address'].fillna('Singapore').tolist()
            address_corpus = [address if address != '' else 'Singapore' for address in address_corpus]
            address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
            address_matrix = address_vectorizer.fit_transform(address_corpus)

            # identify duplicates using ML models
            duplicate_idx = []
            for i in range(len(potential_duplicates)):
                duplicate_idx.append(self._identify_duplicates(potential_duplicates, i, address_matrix))

            assert len(potential_duplicates) == len(duplicate_idx)
            potential_duplicates['duplicates'] = duplicate_idx

            # conflate duplicated POIs
            conflated_pois = self._conflate(potential_duplicates)

            return conflated_pois
        else:
            potential_duplicates

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

            address_similarity = cosine_similarity(address_matrix[neighbour_idx, :],
                                                   address_matrix[centroid_idx, :]).reshape(-1, 1)

            address_str_similarity = np.array(
                [token_set_ratio(data.loc[centroid_idx, 'properties.address'].lower(),
                                 neighbour_address.lower())
                 if not pd.isnull(neighbour_address) else 0.0
                 for neighbour_address
                 in neighbour_data['properties.address'].tolist()]
            ).reshape(-1, 1)

            name_similarity = np.array(
                [token_set_ratio(data.loc[centroid_idx, 'properties.name'].lower(),
                                 neighbour_name.lower())
                 if not pd.isnull(neighbour_name) else 0.0
                 for neighbour_name
                 in neighbour_data['properties.name'].tolist()]
            ).reshape(-1, 1)

            # Pass name and address similarity values into ML models
            predict_prob = np.zeros((len(neighbour_idx), 2))
            for model in self.models:
                predict_prob += model.predict_proba(np.hstack((address_similarity,
                                                               address_str_similarity,
                                                               name_similarity)))
            temp_idx = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
            # POIs are only considered as duplicates if they come from different sources
            duplicate_id = [data.loc[neighbour_idx[idx], 'id'] for idx in temp_idx
                            if data.loc[neighbour_idx[idx], 'properties.source'] !=
                            data.loc[centroid_idx, 'properties.source']]

            # identify duplicates through exact name matches too
            exact_match_id = [data.loc[idx, 'id']
                              for idx in neighbour_idx
                              if ratio(data.loc[centroid_idx, 'properties.name'].lower(),
                                       data.loc[idx, 'properties.name'].lower()) == 100]

            return list(set(duplicate_id + exact_match_id))
        else:  # no neighbours
            return []

    def _conflate(self, data):
        """
        Performs conflation on the duplicated POIs.

        :param data: Geodataframe
            Contains the duplicated POIs.

        :return:
        conflated_pois: Geodataframe
            Contains the conflated POIs.
        """
        conflated_pois = gpd.GeoDataFrame()
        processed_id = []

        for i in range(len(data)):
            if data.loc[i, 'id'] in processed_id:  # ignore POIs that has been merged
                continue

            if len(data.loc[i, 'duplicates']) > 0:
                duplicate_ids, duplicates = self._find_all_duplicates(data.loc[i, 'duplicates'] + [data.loc[i, 'id']],
                                                                      data)
                conflated_pois = conflated_pois.append(self._merge_duplicates(duplicates), ignore_index=True)
                processed_id += duplicate_ids
            else:
                conflated_pois = conflated_pois.append(data.iloc[i], ignore_index=True)
                processed_id.append(data.loc[i, 'id'])

        return conflated_pois

    def _find_all_duplicates(self, duplicate_ids, data):
        """
        Searches for all identified duplicates and return their IDs and attribute information.

        :param duplicate_ids: list
            Contains a list of IDs that are identified to be duplicates.
        :param data: Geodataframe
            Contains the list of neighbouring POIs.

        :return:
        temp_ids: list
            Contains the full list of IDs that are identified as duplicates
        data[data['id'].isin(duplicate_ids)]: Geodataframe:
            Contains the POIs that are identified as duplicates.
        """
        temp_ids = deepcopy(duplicate_ids)
        all_duplicates_not_found = True
        while all_duplicates_not_found:
            temp_data = data[data['id'].isin(temp_ids)].reset_index(drop=True)
            id_list = list(set([item
                                for sublist in temp_data['duplicates'].tolist()
                                for item in sublist] + temp_ids))

            if sorted(id_list) == sorted(temp_ids):
                all_duplicates_not_found = False

            temp_ids = id_list

        return temp_ids, data[data['id'].isin(temp_ids)]

    def _extract_trusted_source_idx(self, data):
        """
        Extract the indices of the POIs that come from the most trusted source.
        OneMap > SLA > GoogleMap > HereMap > OpenstreetMap

        :param data: Geodataframe
            Contain all of the neighbouring POIs in the area.

        :return: list
            Contains the list of indices belonging to POIs from the most trusted source.
        """
        source_list = data['properties.source'].tolist()

        if 'OneMap' in source_list:
            return list(data[data['properties.source'] == 'OneMap'].index)
        elif 'SLA' in source_list:
            return list(data[data['properties.source'] == 'SLA'].index)
        elif 'GoogleMap' in source_list:
            return list(data[data['properties.source'] == 'GoogleMap'].index)
        elif 'HereMap' in source_list:
            return list(data[data['properties.source'] == 'HereMap'].index)
        elif 'OpenStreetMap' in source_list:
            return list(data[data['properties.source'] == 'OpenStreetMap'].index)
        else:
            raise ValueError('{} does not fall into any one of the POI sources.'.format(data['properties.source']))

    def _merge_duplicates(self, duplicates):
        """
        Merges the attributes of all duplicated POIs identified to form a new POI.

        :param duplicates: Geodataframe
            Contains the full list of duplicated POIs.

        :return:
        merged_poi: geoseries
            Contains the merged POI.
        """
        duplicates.reset_index(drop=True, inplace=True)
        trusted_idx = self._extract_trusted_source_idx(duplicates)
        merged_poi = gpd.GeoSeries(crs=4326)
        columns_processed = []

        for column in list(duplicates.columns):
            if column in columns_processed:
                continue

            # feature type
            if column == 'type':
                merged_poi[column] = 'Feature'
                columns_processed.append(column)

            # geometry coordinates
            elif 'geometry' in column:
                if len(duplicates) > 2:
                    centroid = geometry.Polygon([[p.x, p.y] for p in duplicates['geometry'].tolist()]).centroid
                else:
                    centroid = geometry.LineString([[p.x, p.y] for p in duplicates['geometry'].tolist()]).centroid
                merged_poi['geometry.lat'] = centroid.y
                merged_poi['geometry.lng'] = centroid.x
                merged_poi['geometry'] = centroid
                columns_processed += ['geometry', 'geometry.lat', 'geometry.lng']

            # address
            elif column == 'properties.address':
                merged_poi['properties.address'] = max(duplicates.loc[trusted_idx, 'properties.address'].tolist(),
                                                       key=len)
                columns_processed.append(column)

            # name
            elif column == 'properties.name':
                merged_poi['properties.name'] = max(duplicates.loc[trusted_idx, 'properties.name'].tolist(),
                                                    key=len)
                columns_processed.append(column)

            # place type
            elif column == 'properties.place_type':  # store all place types in a list
                temp_list = list(set('; '.join(duplicates[column].tolist()).split('; ')))
                merged_poi[column] = '; '.join(temp_list)
                columns_processed.append(column)

            # tags
            elif 'properties.tags' in column:
                tags = list(set([item for item in duplicates[column].tolist() if not pd.isnull(item)]))
                if len(tags) > 0:
                    merged_poi[column] = '; '.join(tags)
                columns_processed.append(column)

            # source
            elif column == 'properties.source':  # store all sources in a list
                source_list = list(set(duplicates[column].tolist()))
                merged_poi[column] = '; '.join(source_list)
                columns_processed.append(column)

            # requires_verification
            elif 'properties.requires_verification' in column:
                if 'Yes' in duplicates['properties.requires_verification.summary'].tolist():
                    merged_poi['properties.requires_verification.summary'] = 'Yes'
                    merged_poi['properties.requires_verification.reason'] = '; '.join(list(set(
                        [reason for reason in duplicates['properties.requires_verification.reason'].tolist()
                         if not pd.isnull(reason)])))
                else:
                    merged_poi['properties.requires_verification.summary'] = 'No'
                columns_processed += ['properties.requires_verification.summary',
                                      'properties.requires_verification.reason']

            # id information
            elif column == 'id':  # store all ids in a list
                merged_poi[column] = '; '.join(duplicates[column].tolist())
                columns_processed.append(column)

            # stop ids
            elif column == 'stop':
                stop_ids = list(set([item for item in duplicates[column].tolist() if not pd.isnull(item)]))
                if len(stop_ids) > 0:
                    merged_poi[column] = '; '.join(tags)
                columns_processed.append(column)

            # extraction date information
            elif column == 'extraction_date':  # the latest date will be chosen
                extraction_dates = [int(date) for date in duplicates[column].tolist() if not pd.isnull(date)]

                if extraction_dates:
                    merged_poi[column] = str(max(extraction_dates))
                else:
                    raise ValueError('Extraction date information is missing.')
                columns_processed.append(column)

            elif column == 'duplicates':  # ignore duplicate field
                columns_processed.append(column)
                continue

            else:
                raise ValueError('{} is not considered!'.format(column))

        assert set(list(duplicates.columns)).issubset(set(columns_processed))
        return merged_poi


if __name__ == '__main__':
    # extracting POIs based on subzones
    tool = POIConflationTool(subzones=['TAMPINES'])
    tool.conflate_area_poi()
    conflated_data = tool.conflated_data
    osm_data = tool.osm_data
    sla_data = tool.sla_data
    google_data = tool.google_data
    here_data = tool.here_data
    onemap_data = tool.onemap_data

    conflated_data.to_excel('conflated_data.xlsx', index=False)
    osm_data.to_excel('osm_data.xlsx', index=False)
    sla_data.to_excel('sla_data.xlsx', index=False)
    google_data.to_excel('google_data.xlsx', index=False)
    here_data.to_excel('here_data.xlsx', index=False)
    onemap_data.to_excel('onemap_data.xlsx', index=False)


    # extracting POIs on the fly
    # tool = POIConflationTool()
    # data = pd.read_excel('../../data/processed_data/combined_stop_data.xlsx')
    # for i in range(len(data)):
    #     if i < 44081:
    #         continue
    #     print('Processing {}/{}'.format(i+1, len(data)))
    #     tool.extract_poi(data.loc[i, 'StopLat'],
    #                      data.loc[i, 'StopLon'],
    #                      str(data.loc[i, 'StopID']))
