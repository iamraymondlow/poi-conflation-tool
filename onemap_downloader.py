import requests
import json
import time
import os
from util import generate_id, remove_duplicate, extract_date
import pandas as pd
from shapely.geometry import Polygon

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


def extract_query_name(themes):
    """
    Extracts the themes and query terms that are recognised within OneMap's servers.

    :param themes: list
        Contains a list of themes that are of interest.

    :return:
    list(theme_tuple): list
        Contains a list of the themes of interest.
    list(query_tuple): list
        Contains a list of the query terms that corresponds to the themes of interest.
    """
    geocode_url = 'https://developers.onemap.sg/privateapi/themesvc/getAllThemesInfo'
    geocode_url += '?token=' + str(config['onemap_api_key'])

    while True:
        try:
            query_theme = [(theme_dict['THEMENAME'], theme_dict['QUERYNAME'])
                           for theme_dict in requests.get(geocode_url).json()['Theme_Names']
                           if theme_dict['THEMENAME'] in themes]
            theme_tuple, query_tuple = zip(*query_theme)
            return list(theme_tuple), list(query_tuple)

        except requests.exceptions.ConnectionError:
            print('Connection Error. Pausing query for {} minutes...'.format(config['wait_time']))
            time.sleep(config['wait_time'])


def extract_theme(theme):
    """
    This function downloads all point of interest from OneMap's servers that correspond to a particular theme.

    :param theme: string
        A string representing the theme of interest.

    :return:
    requests.get(geocode_url).json(): json
        Contains the downloaded POIs from OneMap in JSON format.
    """
    # Pass query into OneMap API
    geocode_url = 'https://developers.onemap.sg/privateapi/themesvc/retrieveTheme'
    geocode_url += '?queryName=' + theme
    geocode_url += '&token=' + config['onemap_api_key']

    while True:
        try:
            return requests.get(geocode_url).json()

        except json.decoder.JSONDecodeError:
            time.sleep(5)


def extract_address(query_dict):
    """
    Extracts the formatted string address of a POI by concatenating its address substrings.

    :param query_dict: dict
        Contains the address information of a POI split into different components (i.e., block number, street name,
        etc)

    :return:
        formatted_address: str
            Contains the concatenated address string for a particular POI.
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
    Extracts the centroid of a POI that is represented as a polygon.

    :param polygon_coordinates: str
        Contains the coordinates of a POI represented as a polygon.

    :return:
    centroid.y: float
        Contains the latitude of the polygon's centroid.
    centroid.x: float
        Contains the longitude of the polygon's centroid.
    """
    coordinates = polygon_coordinates.split('|')
    bound_coordinates = [(float(latlng.split(',')[1]), float(latlng.split(',')[0])) for latlng in coordinates]
    centroid = Polygon(bound_coordinates).centroid
    return centroid.y, centroid.x


def extract_tags(query_dict):
    """
    Extract the POI's description, address type and building name information as tags.

    :param query_dict: dict
        Contains the POI's description, address type and building name.

    :return:
    tags: dict
        Contains the POI's description, address type and building if they are available.
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
    Perform a mapping of OneMap's theme with Google's place type taxonomy.

    :param theme: str
        Contains OneMap's theme
    :param theme_mapping: dataframe
        Contains the mappings for each theme in OneMap with Google's taxonomy.

    :return:
    mapped_theme: str
        Contains the mapped place type based on Google's taxonomy.
    """
    mapped_theme = theme_mapping[theme_mapping['themes'] == theme]['google_mapping'].tolist()[0]
    return mapped_theme


def format_query_result(query_result, theme, theme_mapping):
    """
    This function takes in the result of the OneMap API and formats it into a JSON format.

    :param query_result: list
        Contains a list of POIs extracted directly from OneMap based on a particular theme.
    :param theme: str
        Contains the theme of the POI.
    :param theme_mapping: dataframe
        Contains the theme mappings for OneMap based on Google's taxonomy.

    :return:
    formatted_query: list
        Contains a list of formatted POIs based on the custom schema.
    """
    formatted_query = []

    if len(query_result) == 0:  # empty result
        return formatted_query

    for i in range(len(query_result)):
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


def main():
    # Extract query name based on selected place types/themes
    theme_mapping = pd.read_excel(config['onemap_mapping'])
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
                print('Connection Error. Pausing query for {} minutes...'.format(config['wait_time']))
                time.sleep(config['wait_time'])

        i += 1

        # load local json file to store query output
        if not os.path.exists(config['onemap_directory']):
            os.makedirs(config['onemap_directory'])

        if os.path.exists(config['onemap_output']):
            with open(config['onemap_output']) as json_file:
                feature_collection = json.load(json_file)
                feature_collection['features'] += format_query_result(query_result['SrchResults'][2:], themes[j],
                                                                      theme_mapping)

            # save query output as json file
            with open(config['onemap_output'], 'w') as json_file:
                json.dump(feature_collection, json_file)

        else:
            with open(config['onemap_output'], 'w') as json_file:
                feature_collection = {'type': 'FeatureCollection',
                                      'features': format_query_result(query_result['SrchResults'][2:], themes[j],
                                                                      theme_mapping)}
                json.dump(feature_collection, json_file)

    # Remove duplicated information
    remove_duplicate(config['onemap_output'])


if __name__ == '__main__':
    main()
