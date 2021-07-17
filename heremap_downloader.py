import requests
import json
import os
import time
import pandas as pd
import geopandas as gpd
from util import extract_date, divide_bounding_box, pixelise_region

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class HereMapScrapper:
    """
    Performs scrapping of nearby POI information from Here Map based on latitude and longitude information.
    """
    def __init__(self, radius):
        self.search_radius = radius

    def extract_area(self, subzones=None):
        """
        Extracts all the POIs in the subzones defined.

        :param subzones: list
            Contains the list of subzones to perform POI scrapping.
        """
        # load country shapefile
        subzones_shp = gpd.read_file(os.path.join(os.path.dirname(__file__), config['country_shapefile']))
        subzones_shp = subzones_shp.to_crs(epsg="4326")
        if subzones is not None:
            subzones_shp = subzones_shp[subzones_shp['PLN_AREA_N'].isin(subzones)].reset_index(drop=True)
            if len(subzones_shp) == 0:
                raise ValueError('Subzone(s) {} is not found in the country shapefiles.'.format(subzones))

        # pixelise region based on shapefile
        coordinate_list = divide_bounding_box(max_lat=config['max_lat'], min_lat=config['min_lat'],
                                              max_lng=config['max_lng'], min_lng=config['min_lng'],
                                              querybox_dim=config['search_radius'])
        coordinate_list = pixelise_region(coordinate_list, subzones_shp)

        # extract POI in the region
        i = 1
        for coordinate in coordinate_list:
            print('Processing query {}/{}'.format(i, len(coordinate_list)))
            lat = (coordinate[2] + coordinate[0]) / 2
            lng = (coordinate[1] + coordinate[3]) / 2
            self._query_poi(lat, lng, query_area=True)
            i += 1

    def extract_poi(self, lat, lng, stop_id):
        """
        Extracts the surrounding POIs near a particular stop either based on cached POI data or making API calls
        on the fly to HERE Map if the stop is encountered for the first time.

        :param lat: float
            Contains the latitude information of the stop.
        :param lng: float
            Contains the longitude information of the stop.
        :param stop_id: str
            Contains the unique ID of the stop.
        :return:
        nearby_pois: GeoDataFrame
            Contains the surrounding POIs found near the stop formatted based on a custom schema.
        """
        # query for nearby POIs using API
        nearby_pois = self._query_poi(lat, lng, stop_id=stop_id)

        # format nearby POIs as geodataframe
        if nearby_pois:
            nearby_pois = pd.json_normalize(nearby_pois)
            nearby_pois = gpd.GeoDataFrame(nearby_pois,
                                           geometry=gpd.points_from_xy(nearby_pois['geometry.lng'],
                                                                       nearby_pois['geometry.lat']))
            return nearby_pois
        else:
            return None

    def _query_poi(self, lat, lng, stop_id=None, query_area=False):
        """
        Performs an API query on the surrounding POIs and caches the resulting POIs.

        :param lat: float
            Contains the latitude information of the stop.
        :param lng: float
            Contains the longitude information of the stop.
        :param stop_id: str
            Contains the unique ID of the stop.
        :param query_area: bool
            Indicates if it is performing an area wide query or point query.

        :return:
        formatted_result: list of dictionary
            Contains the list of neighbouring POIs formatted based on the custom schema.
        """
        not_successful = True
        if query_area:
            cache_directory = config['here_area_cache']
        else:
            cache_directory = config['here_cache']
        while not_successful:
            try:
                query_result = self._perform_query(lat=lat, lng=lng)

                if query_result['results']['items']:
                    formatted_results = self._format_query_result(query_result['results']['items'], stop_id)

                    # store results as cache
                    if not os.path.exists(config['here_directory']):
                        os.makedirs(config['here_directory'])

                    if os.path.exists(cache_directory):  # cache exists
                        with open(cache_directory) as json_file:
                            feature_collection = json.load(json_file)
                            feature_collection['features'] += formatted_results

                        with open(cache_directory, 'w') as json_file:
                            json.dump(feature_collection, json_file)

                    else:  # cache does not exist
                        with open(cache_directory, 'w') as json_file:
                            feature_collection = {'type': 'FeatureCollection',
                                                  'features': formatted_results}
                            json.dump(feature_collection, json_file)

                    return formatted_results

                else:
                    return []

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(config['wait_time']))
                time.sleep(config['wait_time'] * 60)

    def _perform_query(self, lat, lng):
        """
        Extracts all POIs within a bounding circle using HERE Map API.

        :param lat: float
            Contains the latitude information of the stop.
        :param lng: float
            Contains the longitude information of the stop.
        :return:
        requests.get(geocode_url).json(): list
            Contains a list of POIs surrounding the stop.
        """
        # Pass query into HERE API
        geocode_url = 'https://places.ls.hereapi.com/places/v1/discover/explore'
        geocode_url += '?apiKey=' + config['here_api_key']
        geocode_url += '&in=' + str(lat) + ',' + str(lng) + ';r=' + str(self.search_radius)
        geocode_url += '&size' + str(9999)
        geocode_url += '&pretty'

        return requests.get(geocode_url).json()

    def _map_placetype(self, placetype):
        """
        Perform a mapping of HERE Map's categories with Google's place type taxonomy.

        :param placetype: str
            Contains the POI's original place type based on HERE Map's taxonomy.

        :return:
        mapped_placetype[0]: str
            Contains the mapped place type based on Google's taxonomy.
        """
        mapping = pd.read_excel(config['here_mapping'])
        mapped_placetype = mapping[mapping['here_placetype'] == placetype]['google_mapping'].tolist()

        if len(mapped_placetype) == 0:
            return placetype, False
        elif len(mapped_placetype) == 1:
            return mapped_placetype[0], True
        else:
            raise ValueError('More than one mapping is found: {}'.format(mapped_placetype))

    def _format_query_result(self, query_result, stop_id):
        """
        This function takes in the result of the HERE API and formats it into a list of geojson
        dictionary which will be returned. The list will also be saved as a local json file.

        :param query_result: list
            Contains the original query results from HERE API.
        :param stop_id: str
            Contains the ID information of the stop.

        :return:
        poi_data: list
            Contains the formatted query results from HERE API.
        """
        poi_data = []
        for i in range(len(query_result)):
            # extract latitude and longitude information
            lat = query_result[i]['position'][0]
            lng = query_result[i]['position'][1]

            # extract tag information
            if 'tags' in query_result[i].keys():
                tags = query_result[i]['tags'][0]
            else:
                tags = {}

            # perform mapping for place type information
            mapped_placetype, mapping_successful = self._map_placetype(query_result[i]['category']['title'])

            if mapping_successful:
                verification = {'summary': 'No'}
            else:
                verification = {'summary': 'Yes', 'reason': 'Mapping not found'}

            poi_dict = {
                'type': 'Feature',
                'geometry': {'lat': lat, 'lng': lng},
                'properties': {'address': query_result[i]['vicinity'].replace('<br/>', ' '),
                               'name': query_result[i]['title'],
                               'place_type': mapped_placetype,
                               'tags': tags,
                               'source': 'HereMap',
                               'requires_verification': verification},
                'stop': stop_id,
                'id': str(query_result[i]['id']),
                'extraction_date': extract_date()
            }
            poi_data.append(poi_dict)

        return poi_data


if __name__ == '__main__':
    scrapper = HereMapScrapper(config['search_radius'])
    scrapper.extract_area(subzones=['PUNGGOL', 'QUEENSTOWN'])
