import requests
import json
import os.path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from util import remove_duplicate, extract_date, capitalise_string
from tqdm import tqdm
from shapely.geometry import Point

here_app_key = 'ChgzzPNIMr-lHVXDqgEFpuV9HbOwLzcB5SCxHpy_l8s'
onemap_api = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOjMyMTYsInVzZXJfaWQiOjMyMTYsImVtYWlsIjoiaWFtcmF5bW9uZGxvd0BnbWFpbC5jb20iLCJmb3JldmVyIjpmYWxzZSwiaXNzIjoiaHR0cDpcL1wvb20yLmRmZS5vbmVtYXAuc2dcL2FwaVwvdjJcL3VzZXJcL3Nlc3Npb24iLCJpYXQiOjE2MTgzMjM2OTcsImV4cCI6MTYxODc1NTY5NywibmJmIjoxNjE4MzIzNjk3LCJqdGkiOiI0OWRkMmEwOTM4YmM4NGZhOTVkNmRkNjAxYjk3MTU2ZiJ9.M6mLLCpBhXzbygvpW59ME4I-ZGCcbI5BHEzmbH6CKZk'
wait_time = 5
output_filename = 'data/osm/osm_poi.json'
osm_filenames = ['gis_osm_pofw_a_free_1.shp', 'gis_osm_pois_a_free_1.shp', 'gis_osm_traffic_a_free_1.shp']
search_radius = 20


def query_address(lat, lng):
    """
    Perform reverse geocoding using the POI's latitude and longitude information to obtain address information from
    HERE Maps.
    """
    # Pass query into Onemap for reverse geocoding
    geocode_url = 'https://developers.onemap.sg/privateapi/commonsvc/revgeocode?location={},{}'.format(lat, lng)
    geocode_url += '&token={}'.format(onemap_api)
    geocode_url += '&buffer={}'.format(search_radius)
    geocode_url += '&addressType=all'

    while True:
        try:
            query_result = requests.get(geocode_url).json()
            address = ''
            # take the address from the first query result in the list
            if 'BLOCK' in query_result['GeocodeInfo'][0]:
                address += query_result['GeocodeInfo'][0]['BLOCK'] + ' '
            if 'ROAD' in query_result['GeocodeInfo'][0]:
                address += query_result['GeocodeInfo'][0]['ROAD'] + ' '
            if 'POSTALCODE' in query_result['GeocodeInfo'][0]:
                address += 'Singapore ' + query_result['GeocodeInfo'][0]['POSTALCODE'] + ' '

            address = capitalise_string(address[:-1])
            return address

        except requests.exceptions.ConnectionError:
            print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
            time.sleep(wait_time * 60)

        except:
            return None


def perform_mapping(place_type):
    """
    Performs mapping of OSM's place type to Google's taxonomy.
    """
    placetype_mapping = pd.read_excel('data/mappings/osm_mapping.xlsx')
    placetype_list = placetype_mapping[placetype_mapping['osm_placetype'] == place_type]['google_mapping'].tolist()
    if len(placetype_list) == 0:
        return place_type, False
    elif len(placetype_list) == 1:
        return placetype_list[0], True
    else:
        raise ValueError('More than one mapping returned for place type {}: {}'.format(place_type, placetype_list))


def format_poi(poi):
    """
    Formats the POI into a JSON format.
    """
    # Extract geometry information
    if poi.geometry.geom_type == 'Point':
        geometry = {'lat': poi.geometry.y, 'lng': poi.geometry.x}
    elif poi.geometry.geom_type == 'Polygon' or poi.geometry.geom_type == 'MultiPolygon':
        geometry = {'lat': poi.geometry.centroid.y, 'lng': poi.geometry.centroid.x}
    else:
        raise ValueError('{} is not supported'.format(poi.geometry.geom_type))

    # Extract osm address from here map
    address = query_address(geometry['lat'], geometry['lng'])

    # Extract place type
    place_type, mapping_successful = perform_mapping(poi['fclass'])

    if mapping_successful:
        verification = {'summary': 'No'}
    else:
        verification = {'summary': 'Yes', 'reason': 'Mapping not found'}

    poi_dict = {
        'type': 'Feature',
        'geometry': geometry,
        'properties': {'address': address,
                       'name': poi['name'],
                       'place_type': place_type,
                       'source': 'OpenStreetMap',
                       'requires_verification': verification},
        'id': str(poi['osm_id']),
        'extraction_date': extract_date()
    }

    return poi_dict


def within_boundary(poi, country_shapefile):
    """
    Check if the POI fall within the study area.
    """
    if poi.geometry is None:  # ignore data point if it does not have geometry information
        return False

    if poi.geometry.geom_type == 'Point':
        num_within = int(np.sum(country_shapefile['geometry'].apply(lambda x: poi.geometry.within(x))))
    elif poi.geometry.geom_type == 'Polygon' or poi.geometry.geom_type == 'MultiPolygon':
        num_within = int(np.sum(country_shapefile['geometry']
                                .apply(lambda x: Point(poi.geometry.centroid.x, poi.geometry.centroid.y).within(x))))
    else:
        raise ValueError('{} is not supported'.format(poi.geometry.geom_type))

    assert num_within <= 1
    if num_within == 0:
        return False
    else:
        return True

def process_osm():
    # Import shapefile for Singapore
    country_shapefile = gpd.read_file('data/country_shapefiles/MP14_REGION_NO_SEA_PL.shp')
    country_shapefile = country_shapefile.to_crs(epsg='4326')

    # Import shape file for OSM POI data
    for filename in osm_filenames:
        poi_shp = gpd.read_file('data/osm/{}'.format(filename))
        poi_shp = poi_shp.to_crs(epsg="4326")

        # format POI data
        print('Processing {}...'.format(filename))
        for i in tqdm(range(len(poi_shp))):
            if within_boundary(poi_shp.iloc[i], country_shapefile):
                formatted_poi = format_poi(poi_shp.iloc[i])

                # save formatted POI data locally
                if os.path.exists(output_filename):
                    with open(output_filename) as json_file:
                        feature_collection = json.load(json_file)
                        feature_collection['features'].append(formatted_poi)

                    with open(output_filename, 'w') as json_file:
                        json.dump(feature_collection, json_file)
                else:
                    with open(output_filename, 'w') as json_file:
                        feature_collection = {'type': 'FeatureCollection', 'features': [formatted_poi]}
                        json.dump(feature_collection, json_file)
            else:
                continue

    # Remove duplicated information
    remove_duplicate(output_filename)


if __name__ == '__main__':
    process_osm()