import requests
import json
import pandas as pd
import os.path
import time
from util import remove_duplicate, extract_date

# load config file
with open('config.json') as f:
    config = json.load(f)


class HereMapScrapper:
    """
    Performs scrapping of nearby POI information from Here Map based on latitude and longitude information.
    """
    def __init__(self, radius):
        self.search_radius = radius

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
        dict
            Contains the surrounding POIs found near the stop formatted based on a custom schema.
        """
        if os.path.exists(config['here_cache']):  # check if cache exist
            with open(config['here_cache']) as json_file:
                feature_collection = json.load(json_file)

            # check if cache contains the POI for this stop
            filtered_features = [item for item in feature_collection['features'] if item['stop'] == stop_id]

            if len(filtered_features) > 0:  # cache contains POIs for this stop
                return {"type": "FeatureCollection", "features": filtered_features}

            else:  # cache does not contain POIs for this stop
                filtered_features = self._query_poi(lat, lng, stop_id)

                return {"type": "FeatureCollection", "features": filtered_features}

        else:  # cache does not exist
            filtered_features = self._query_poi(lat, lng, stop_id)

            return {"type": "FeatureCollection", "features": filtered_features}

    def _query_poi(self, lat, lng, stop_id):
        """
        Performs API query on the surrounding POIs

        :param lat: float
            Contains the latitude information of the stop.
        :param lng: float
            Contains the longitude information of the stop.
        :param stop_id: str
            Contains the unique ID of the stop.
        :return:
        """
        not_successful = True
        while not_successful:
            try:
                query_result = self._perform_query(lat=lat, lng=lng)

                if query_result['results']['items']:
                    formatted_results = self._format_query_result(query_result['results']['items'], stop_id)

                    # store results as cache
                    if not os.path.exists(config['here_directory']):
                        os.makedirs(config['here_directory'])

                    if os.path.exists(config['here_cache']):  # cache exists
                        with open(config['here_cache']) as json_file:
                            feature_collection = json.load(json_file)
                            feature_collection['features'] += formatted_results

                        with open(config['here_cache'], 'w') as json_file:
                            json.dump(feature_collection, json_file)

                    else:  # cache does not exist
                        with open(config['here_cache'], 'w') as json_file:
                            feature_collection = {'type': 'FeatureCollection',
                                                  'features': formatted_results}
                            json.dump(feature_collection, json_file)

                    # Removing duplicate data
                    remove_duplicate(config['here_cache'])

                    return formatted_results

                else:
                    return []

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(config['wait_time']))
                time.sleep(config['wait_time'] * 60)

            except ValueError:
                print('Pausing query for {} minutes...'.format(config['wait_time']))
                time.sleep(config['wait_time'] * 60)

    def _perform_query(self, lat, lng):
        """
        Extracts all POIs within a bounding circle using HERE Map API.

        :param lat: float
            Contains the latitude information of the stop.
        :param lng: float
            Contains the longitude information of the stop.
        :return:
        list
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
