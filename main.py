import json
import os
import model
from onemap_downloader import OneMap
from osm_processor import OSM
from sla_processor import SLA

# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class POIConflationTool:
    """
    Performs POI conflation based on 5 different data sources (i.e., OSM, OneMap, SLA, Google Maps
    and HERE Map).
    """
    def __init__(self):
        """
        Checks if the OSM, OneMap, and SLA datasets are formatted. If any of the datasets are
        not formatted, the appropriate functions will be triggered to begin formatting the dataset.
        Also checks if machine learning model for identifying POI duplicates are trained.
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

        # check if machine learning model is trained. If not, train model.
        if not os.listdir(os.path.exists(os.path.join(os.path.dirname(__file__), config['models_directory']))):
            model.train_model()

        # load local POI data and machine learning models
        self.osm_data = None
        self.onemap_data = None
        self.sla_data = None
        self.google_data = None
        self.heremap_data = None
        self.models = None

    def extract_poi(self, lat, lng):
        """
        Extracts the neighbouring POIs based on the geograhical coordinates and performs POI conflation.
        """
        # extracts neighbouring POIs from OSM

        # extract neighbouring POIs from OneMap

        # extract neighbouring POIs from SLA

        # query neighbouring POIs from GoogleMap

        # query neighbouring POIs from HERE Map

        # perform conflation

        return None

    def _perform_conflation(self):
        return None





