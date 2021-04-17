import json
import pandas as pd
import geopandas as gpd
from util import remove_duplicate, extract_date, capitalise_string
from pyproj import Proj

trade_codes = ['9ANSCH', '9ATM', '9CC', '9CCARE', '9CDEF', '9CENI', '9CHNTE', '9CHU', '9CLNI', '9COT',
               '9FF', '9FLSCH', '9FSSCH', '9GNS', '9HDBBT', '9HEC', '9HOSP', '9HOSPI', '9HOT',
               '9INDTE', '9INSEC', '9ITE', '9JC', '9KG', '9LBH', '9LIB', '9MOS', '9NPC', '9OTHIN',
               '9PBCOM', '9PINT', '9PO', '9POL', '9POLY', '9PRI', '9PTL', '9RCLUB', '9RESCH',
               '9SCARE', '9SCTRE', '9SEC', '9SHTEM', '9SPSCH', '9SPT', '9SWC', '9SYNA', '9TCH',
               '9TI', '9VET', '9VI', '19BDPT', '19BINT', '19BTER']
trade_types = ['H', 'B']
data_types = ['EXTG', 'UC', 'PROP']
output_filename = 'data/sla/sla_poi.json'


def perform_mapping(gpd_file):
    """
    Maps the terms in the TRADE_CODE, TRADE_TYPE and DATA_TYPE fields into its human readable form.
    """
    placetype_mapping = pd.read_excel('data/mappings/sla_mapping.xlsx')
    abbreviation_list = placetype_mapping['trade_code'].tolist()

    # Perform mapping for TRADE_CODE
    for trade_code in trade_codes:
        tradecode_index = abbreviation_list.index(trade_code)
        index_list = gpd_file.index[gpd_file['TRADE_CODE'] == trade_code].tolist()
        gpd_file.loc[index_list, 'TRADE_CODE'] = placetype_mapping.loc[tradecode_index, 'google_mapping']

    # Perform mapping for TRADE_TYPE
    for trade_type in trade_types:
        tradetype_index = abbreviation_list.index(trade_type)
        index_list = gpd_file.index[gpd_file['TRADE_TYPE'] == trade_type].tolist()
        gpd_file.loc[index_list, 'TRADE_TYPE'] = placetype_mapping.loc[tradetype_index, 'sla_placetype']

    # Perform mapping for DATA_TYPE
    for data_type in data_types:
        datatype_index = abbreviation_list.index(data_type)
        index_list = gpd_file.index[gpd_file['DATA_TYPE_'] == data_type].tolist()
        gpd_file.loc[index_list, 'DATA_TYPE_'] = placetype_mapping.loc[datatype_index, 'sla_placetype']

    return gpd_file


def format_address(gpd_row):
    """
    Arrange the various address components into its respective fields in a dictionary format.
    """
    formatted_address = ''

    if pd.notnull(gpd_row['HOUSE_BLK_']):
        formatted_address += str(gpd_row['HOUSE_BLK_']) + ' '

    if pd.notnull(gpd_row['ROAD_NAME']):
        formatted_address += capitalise_string(gpd_row['ROAD_NAME']) + ' '

    if pd.notnull(gpd_row['LEVEL_NO']):
        formatted_address += 'Level {} '.format(gpd_row['LEVEL_NO'])

    if pd.notnull(gpd_row['UNIT_NO']):
        formatted_address += 'Unit {} '.format(gpd_row['UNIT_NO'])

    if pd.notnull(gpd_row['POSTAL_CD']):
        formatted_address += 'Singapore ' + str(gpd_row['POSTAL_CD']) + ' '

    return formatted_address[:-1]


def extract_tags(gpd_row):
    """
    Extract the trade brand, trade type and data type information as tag information in the POI.
    """
    tags = {}

    if pd.notna(gpd_row['TRADE_BRAN']):
        tags['parent'] = capitalise_string(gpd_row['TRADE_BRAN'])

    if pd.notna(gpd_row['TRADE_TYPE']):
        tags['trade_type'] = gpd_row['TRADE_TYPE']

    if pd.notna(gpd_row['DATA_TYPE_']):
        tags['data_type'] = gpd_row['DATA_TYPE_']

    return tags


def format_features(data):
    """
    Formats the POI features based on the custom schema.
    """
    features = []
    for i in range(len(data)):
        poi_dict = {'type': 'Feature',
                    'geometry': {'lat': data.loc[i, 'LAT'], 'lng': data.loc[i, 'LNG']},
                    'properties': {'address': format_address(data.loc[i, :]),
                                   'name': capitalise_string(data.loc[i, 'TRADE_NAME']),
                                   'place_type': data.loc[i, 'TRADE_CODE'],
                                   'tags': extract_tags(data.loc[i, :]),
                                   'source': 'SLA',
                                   'requires_verification': {'summary': 'No'}},
                    'id': str(data.loc[i, 'OBJECTID']),
                    'extraction_date': extract_date()
                    }
        features.append(poi_dict)

    return features


def process_sla():
    # import shape files
    data = None
    for trade_code in trade_codes:
        if data is None:
            data = gpd.read_file('data/sla/{}.shp'.format(trade_code))
        else:
            data = data.append(gpd.read_file('data/sla/{}.shp'.format(trade_code)))
    data.reset_index(drop=True, inplace=True)

    # Perform abbreviation mapping
    data = perform_mapping(data)

    # Extract latitude longitude information
    proj = Proj(data.crs)
    lng_lat = [proj(geometry.x, geometry.y, inverse=True) for geometry in data['geometry']]
    lng, lat = zip(*lng_lat)
    data['LAT'] = lat
    data['LNG'] = lng

    # Transform into JSON format and save on local directory
    with open(output_filename, 'w') as json_file:
        feature_collection = {'type': 'FeatureCollection',
                              'features': format_features(data)}
        json.dump(feature_collection, json_file)

    # Remove duplicated information
    remove_duplicate(output_filename)


if __name__ == '__main__':
    process_sla()
