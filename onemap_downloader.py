import requests
import json
import time
import os
from util import generate_id, remove_duplicate, extract_date
import pandas as pd
from shapely.geometry import Polygon

# load config file


# load parameters
token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOjMyMTYsInVzZXJfaWQiOjMyMTYsImVtYWlsIjoiaWFtcmF5bW9uZGxvd0BnbWFpbC5jb20iLCJmb3JldmVyIjpmYWxzZSwiaXNzIjoiaHR0cDpcL1wvb20yLmRmZS5vbmVtYXAuc2dcL2FwaVwvdjJcL3VzZXJcL3Nlc3Npb24iLCJpYXQiOjE2MTgzMjM2OTcsImV4cCI6MTYxODc1NTY5NywibmJmIjoxNjE4MzIzNjk3LCJqdGkiOiI0OWRkMmEwOTM4YmM4NGZhOTVkNmRkNjAxYjk3MTU2ZiJ9.M6mLLCpBhXzbygvpW59ME4I-ZGCcbI5BHEzmbH6CKZk'
wait_time = 15  # sets the number of minutes to wait between each query when your API limit is reached
output_filename = 'data/onemap/onemap_poi.json'
file_directory = 'data/onemap'


def extract_query_name(themes):
    """

    :param themes:
    :return:
    """
    geocode_url = 'https://developers.onemap.sg/privateapi/themesvc/getAllThemesInfo'
    geocode_url += '?token=' + str(token)

    while True:
        try:
            query_theme = [(theme_dict['THEMENAME'], theme_dict['QUERYNAME'])
                           for theme_dict in requests.get(geocode_url).json()['Theme_Names']
                           if theme_dict['THEMENAME'] in themes]
            theme_tuple, query_tuple = zip(*query_theme)
            return list(theme_tuple), list(query_tuple)

        except requests.exceptions.ConnectionError:
            print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
            time.sleep(wait_time)


def extract_theme(theme):
    """
    This function extracts all locations related to a particular theme.

    :param theme:

    :return:
    """
    # Pass query into OneMap API
    geocode_url = 'https://developers.onemap.sg/privateapi/themesvc/retrieveTheme'
    geocode_url += '?queryName=' + theme
    geocode_url += '&token=' + token

    while True:
        try:
            return requests.get(geocode_url).json()

        except json.decoder.JSONDecodeError:
            time.sleep(5)


def extract_address(query_dict):
    """
    Extracts the formatted address information of a POI by concatenating its address substrings.
    """
    formatted_address = ''
    if 'ADDRESSBLOCKHOUSENUMBER' in query_dict.keys() and query_dict['ADDRESSBLOCKHOUSENUMBER'] != 'null':
        formatted_address += query_dict['ADDRESSBLOCKHOUSENUMBER'] + ' '

    if 'ADDRESSSTREETNAME' in query_dict.keys() and query_dict['ADDRESSSTREETNAME'] != 'null':
        formatted_address += query_dict['ADDRESSSTREETNAME'] + ' '

    if 'ADDRESSUNITNUMBER' in query_dict.keys() and query_dict['ADDRESSUNITNUMBER'] != 'null':
        formatted_address += 'Unit ' + query_dict['ADDRESSUNITNUMBER'] + ' '

    if 'ADDRESSFLOORNUMBER' in query_dict.keys() and query_dict['ADDRESSFLOORNUMBER'] != 'null':
        formatted_address += 'Level ' + query_dict['ADDRESSFLOORNUMBER'] + ' '

    if 'ADDRESSPOSTALCODE' in query_dict.keys() and query_dict['ADDRESSPOSTALCODE'] != 'null':
        formatted_address += 'Singapore ' + query_dict['ADDRESSPOSTALCODE'] + ' '
    return formatted_address[:-1]


def extract_polygon_centroid(polygon_coordinates):
    """
    Extract the centroid of a POI that is represented as a polygon.
    """
    coordinates = polygon_coordinates.split('|')
    bound_coordinates = [(float(latlng.split(',')[1]), float(latlng.split(',')[0])) for latlng in coordinates]
    centroid = Polygon(bound_coordinates).centroid
    return centroid.y, centroid.x


def extract_tags(query_dict):
    """
    Extract the POI's description, address type and building name information as tags.
    """
    tags = {}
    if "DESCRIPTION" in query_dict.keys() and query_dict['DESCRIPTION'] != 'null':
        tags.update({'description': query_dict['DESCRIPTION']})

    if 'ADDRESSTYPE' in query_dict.keys() and query_dict['ADDRESSTYPE'] != 'null':
        tags.update({'address_type': query_dict['ADDRESSTYPE']})

    if 'ADDRESSBUILDINGNAME' in query_dict.keys() and query_dict['ADDRESSBUILDINGNAME'] != 'null':
        tags.update({'building_name': query_dict['ADDRESSBUILDINGNAME']})

    return tags


def map_placetype(theme, theme_mapping):
    """
    Perform a mapping of the theme with Google's place type taxonomy.
    """
    mapped_theme = theme_mapping[theme_mapping['themes'] == theme]['google_mapping'].tolist()[0]
    return mapped_theme


def format_query_result(query_result, theme, theme_mapping):
    """
    This function takes in the result of the OneMap API and formats it into a JSON format.
    """
    formatted_query = []

    if len(query_result) == 0:  # empty result
        return formatted_query

    for i in range(len(query_result)):
        bound_coordinates = None
        if '|' in query_result[i]['LatLng']:
            lat, lng = extract_polygon_centroid(query_result[i]['LatLng'])
        else:
            lat, lng = [float(item) for item in query_result[i]['LatLng'].split(',')]

        formatted_address = extract_address(query_result[i])

        poi_dict = {'type': 'Feature',
                    'geometry': {'lat': lat, 'lng': lng},
                    'properties': {'address': formatted_address,
                                   'name': query_result[i]['NAME'],
                                   'place_type': map_placetype(theme, theme_mapping),
                                   'tags': extract_tags(query_result[i]),
                                   'source': 'OneMap',
                                   'requires_verification': {'summary': 'No'}}}

        poi_dict['id'] = str(generate_id(poi_dict))
        poi_dict['extraction_date'] = extract_date()

        formatted_query.append(poi_dict)

    return formatted_query


def download_data():
    # Extract query name based on selected place types/themes
    theme_mapping = pd.read_excel('data/mappings/onemap_mapping.xlsx')
    themes, query_names = extract_query_name(theme_mapping['themes'].to_list())
    assert len(themes) == len(query_names)

    # Extract POI information based on selected place types/themes
    i = 1
    for j in range(len(themes)):
        print('Extracting {}...{}/{} themes'.format(themes[j], i, len(themes)))

        not_successful = True
        while not_successful:
            try:
                query_result = extract_theme(query_names[j])
                not_successful = False

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
                time.sleep(wait_time)

        i += 1

        # load local json file to store query output
        if not os.path.exists('data/onemap'):
            os.makedirs('data/onemap')

        if os.path.exists(output_filename):
            with open(output_filename) as json_file:
                feature_collection = json.load(json_file)
                feature_collection['features'] += format_query_result(query_result['SrchResults'][2:], themes[j],
                                                                      theme_mapping)

            # save query output as json file
            with open(output_filename, 'w') as json_file:
                json.dump(feature_collection, json_file)

        else:
            with open(output_filename, 'w') as json_file:
                feature_collection = {'type': 'FeatureCollection',
                                      'features': format_query_result(query_result['SrchResults'][2:], themes[j],
                                                                      theme_mapping)}
                json.dump(feature_collection, json_file)

    # Remove duplicated information
    remove_duplicate(output_filename)


if __name__ == '__main__':
    download_data()
