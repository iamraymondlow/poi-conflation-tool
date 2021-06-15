import requests
import json
import os.path
import time
from util import remove_duplicate, extract_date

# load config file
with open('config.json') as f:
    config = json.load(f)


class GoogleMapScrapper:
    """
    Performs scrapping of nearby POI information from Google Map based on latitude and longitude information.
    """
    def __init__(self, radius):
        self.search_radius = radius

    def extract_poi(self, lat, lng, stop_id):
        """
        Extracts the surrounding POIs near a particular stop either based on cached POI data or making API calls
        on the fly to Google Places if the stop is encountered for the first time.

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
        if os.path.exists(config['google_cache']):  # check if cache exist
            with open(config['google_cache']) as json_file:
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
        Performs an API query on the surrounding POIs and caches the resulting POIs.

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

                if query_result['results']:
                    formatted_results = self._format_query_result(query_result['results'], stop_id)

                    # extract the other POIs stored in the next page
                    while 'next_page_token' in query_result:
                        query_result = self._perform_query(next_page_token=query_result['next_page_token'])
                        formatted_results += self._format_query_result(query_result['results'], stop_id)

                    # store results as cache
                    if not os.path.exists(config['google_directory']):
                        os.makedirs(config['google_directory'])

                    if os.path.exists(config['google_cache']):  # cache exists
                        with open(config['google_cache']) as json_file:
                            feature_collection = json.load(json_file)
                            feature_collection['features'] += formatted_results

                        with open(config['google_cache'], 'w') as json_file:
                            json.dump(feature_collection, json_file)

                    else:  # cache does not exist
                        with open(config['google_cache'], 'w') as json_file:
                            feature_collection = {'type': 'FeatureCollection',
                                                  'features': formatted_results}
                            json.dump(feature_collection, json_file)

                    # Removing duplicate data
                    remove_duplicate(config['google_cache'])

                    return formatted_results

                else:
                    return []

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(config['wait_time']))
                time.sleep(config['wait_time'] * 60)

            except ValueError:
                print('Pausing query for {} minutes...'.format(config['wait_time']))
                time.sleep(config['wait_time'] * 60)

    def _perform_query(self, lat=None, lng=None, next_page_token=None):
        """
        Extracts all POIs within a bounding circle using the Google Places API.

        :param lat: float
            Contains the latitude information of the stop.
        :param lng: float
            Contains the longitude information of the stop.
        :param next_page_token: str
            Contains the page token information for queries that have more than 20 results.

        :return:
        query_result.json(): list
            Contains a list of POIs surrounding the stop.
        """
        # Pass query into Google Places API
        geocode_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'

        if (lat is not None) and (lng is not None):
            params = dict(key=config['google_api_key'],
                          location=str(lat)+','+str(lng),
                          radius=str(self.search_radius))

        elif next_page_token is not None:
            time.sleep(3)  # sleep for a few seconds for the page token to become valid
            params = dict(key=config['google_api_key'],
                          pagetoken=next_page_token)

        else:
            raise ValueError('User must either provide the lat/lng information or next_page_token information')

        query_result = requests.get(url=geocode_url, params=params)

        return query_result.json()

    def _extract_name(self, query_result):
        """
        Extracts the name information for a neighbouring POI. In the case, where the name information
        is not available, an empty string is returned instead.

        :param query_result: dict
            Contains the information for a particular POI.

        :return:
        str:
            Contains the name information for a particular POI.
        """
        if 'name' in query_result.keys():
            return query_result['name']
        else:
            return ''

    def _concat_placetype(self, placetype_list):
        """
        Concatenates the list of place types into a single string.

        :param placetype_list: list
            Contains the list of place types from Google Places.

        :return:
        combined_placetype[:-2]: str
            Contains the concatenated place types in a single string.
        """
        combined_placetype = ''
        for placetype in placetype_list:
            combined_placetype += placetype + '; '
        return combined_placetype[:-2]

    def _format_query_result(self, query_result, stop_id):
        """
        This function takes in the result of the Google Map API and formats it based on a custom schema.

        :param query_result: dict
            Contains the original attributes of a POI from Google Places.
        :param stop_id: str
            Contains the unique ID of the stop.

        :return:
        poi_data: list
            Contains a list of the formatted POIs based on the custom schema.
        """
        poi_data = []
        for i in range(len(query_result)):
            # extract latitude and longitude information
            lat = query_result[i]['geometry']['location']['lat']
            lng = query_result[i]['geometry']['location']['lng']

            # remove irrelevant place types
            ignored_placetypes = ['route', 'neighborhood']
            if bool(set(ignored_placetypes) & set(query_result[i]['types'])):
                continue

            # extract address information
            if 'vicinity' in query_result[i].keys():
                address = query_result[i]['vicinity']
            else:
                address = 'Singapore'

            poi_dict = {
                'type': 'Feature',
                'geometry': {'lat': lat, 'lng': lng},
                'properties': {'address': address,
                               'name': self._extract_name(query_result[i]),
                               'place_type': self._concat_placetype(query_result[i]['types']),
                               'source': 'GoogleMap',
                               'requires_verification': {'summary': 'No'}},
                'stop': stop_id,
                'id': str(query_result[i]['place_id']),
                'extraction_date': extract_date()
            }
            poi_data.append(poi_dict)

        return poi_data
