import json
import os
from onemap_downloader import OneMap
from osm_processor import OSM
from sla_processor import SLA

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class Setup:
    """
    This class is used to setup the POI conflation tool by formatting the OSM, OneMap, and SLA datasets
    based on the custom schema.
    """
    def __init__(self):
        """
        Checks if the OSM, OneMap, and SLA datasets are formatted. If any of the datasets are
        not formatted, the appropriate functions will be triggered to begin formatting the dataset.
        """
        # check if OneMap data is formatted. If not, format OneMap data.
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['onemap_output'])):
            OneMap().format_data()

        # check if SLA data is formatted. If not, format SLA data.
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['sla_output'])):
            SLA().format_data()

        # check if OSM data is formatted. If not, format OSM data.
        if not os.path.exists(os.path.join(os.path.dirname(__file__), config['osm_output'])):
            OSM().format_data()
