import json
import pandas as pd
import geopandas as gpd
import os
from util import remove_duplicate, extract_date, capitalise_string
from pyproj import Proj

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class SLA:
    def format_data(self):
        """
        Formats the SLA dataset into a custom schema and saves it locally.
        """
        # import shape files
        data = None
        for trade_code in config['sla_trade_codes']:
            if data is None:
                data = gpd.read_file(config['sla_data_directory'].format(trade_code))
            else:
                data = data.append(gpd.read_file(config['sla_data_directory'].format(trade_code)))
        data.reset_index(drop=True, inplace=True)

        # Perform abbreviation mapping
        data = self._perform_mapping(data)

        # Extract latitude longitude information
        proj = Proj(data.crs)
        lng_lat = [proj(geometry.x, geometry.y, inverse=True) for geometry in data['geometry']]
        lng, lat = zip(*lng_lat)
        data['LAT'] = lat
        data['LNG'] = lng

        # Transform into JSON format and save on local directory
        with open(config['sla_output'], 'w') as json_file:
            feature_collection = {'type': 'FeatureCollection',
                                  'features': self._format_features(data)}
            json.dump(feature_collection, json_file)

        # Remove duplicated information
        remove_duplicate(config['sla_output'])

    def _perform_mapping(self, gpd_file):
        """
        Maps the terms in the TRADE_CODE, TRADE_TYPE and DATA_TYPE fields into its human readable form.

        :param gpd_file: geopandas dataframe
            Contains the SLA dataset in its raw format.

        :return:
        gpd_file: geopandas dataframe
            Contains the SLA dataset with its TRADE_CODE, TRADE_TYPE, and DATA_TYPE information mapped to its
            human readable form.
        """
        placetype_mapping = pd.read_excel(config['sla_mapping'])
        abbreviation_list = placetype_mapping['trade_code'].tolist()

        # Perform mapping for TRADE_CODE
        for trade_code in config['sla_trade_codes']:
            tradecode_index = abbreviation_list.index(trade_code)
            index_list = gpd_file.index[gpd_file['TRADE_CODE'] == trade_code].tolist()
            gpd_file.loc[index_list, 'TRADE_CODE'] = placetype_mapping.loc[tradecode_index, 'google_mapping']

        # Perform mapping for TRADE_TYPE
        for trade_type in config['sla_trade_types']:
            tradetype_index = abbreviation_list.index(trade_type)
            index_list = gpd_file.index[gpd_file['TRADE_TYPE'] == trade_type].tolist()
            gpd_file.loc[index_list, 'TRADE_TYPE'] = placetype_mapping.loc[tradetype_index, 'sla_placetype']

        # Perform mapping for DATA_TYPE
        for data_type in config['sla_data_types']:
            datatype_index = abbreviation_list.index(data_type)
            index_list = gpd_file.index[gpd_file['DATA_TYPE_'] == data_type].tolist()
            gpd_file.loc[index_list, 'DATA_TYPE_'] = placetype_mapping.loc[datatype_index, 'sla_placetype']

        return gpd_file

    def _format_address(self, gpd_row):
        """
        Arrange the various address components into its respective fields in a dictionary format.

        :param gpd_row: geopandas series
            Contains the unformatted POI information from SLA dataset with address information.

        :return:
        formatted_address: str
            Contains the formatted address of a particular POI
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

    def _extract_tags(self, gpd_row):
        """
        Extract the trade brand, trade type and data type information as tag information in the POI.

        :param gpd_row: geopandas series
            Contains the POI information from the SLA dataset.

        :return:
        tags: dict
            A dictionary of different tags assigned to the POI describing its parent entity, trade type and data type.
        """
        tags = {}

        if pd.notna(gpd_row['TRADE_BRAN']):
            tags['parent'] = capitalise_string(gpd_row['TRADE_BRAN'])

        if pd.notna(gpd_row['TRADE_TYPE']):
            tags['trade_type'] = gpd_row['TRADE_TYPE']

        if pd.notna(gpd_row['DATA_TYPE_']):
            tags['data_type'] = gpd_row['DATA_TYPE_']

        return tags

    def _format_features(self, data):
        """
        Formats the POI features based on the custom schema.

        :param data: geopandas dataframe
            Contains the unformatted POI data from SLA.

        :return:
        features: list
           Contains the formatted POI data formatted based on the custom schema.
        """
        features = []
        for i in range(len(data)):
            poi_dict = {'type': 'Feature',
                        'geometry': {'lat': data.loc[i, 'LAT'], 'lng': data.loc[i, 'LNG']},
                        'properties': {'address': self._format_address(data.loc[i, :]),
                                       'name': capitalise_string(data.loc[i, 'TRADE_NAME']),
                                       'place_type': data.loc[i, 'TRADE_CODE'],
                                       'tags': self._extract_tags(data.loc[i, :]),
                                       'source': 'SLA',
                                       'requires_verification': {'summary': 'No'}},
                        'id': str(data.loc[i, 'OBJECTID']),
                        'extraction_date': extract_date()
                        }
            features.append(poi_dict)

        return features
