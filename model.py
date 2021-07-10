import pandas as pd
import os
import json
import re


# load config file
with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.load(f)


class Model:
    """
    This class perform model training for identifying duplicated POIs between different data sources.
    """
    def train_model(self):
        """
        This function performs model training for identifying duplicated POIs between different data
        sources.
        """
        # load labeled data
        labeled_data = pd.read_csv(os.path.join(os.path.dirname(__file__), config['labeled_data']))
        labeled_data['duplicates'] = labeled_data['duplicates'].apply(self._format_duplicates)
        labeled_data = labeled_data[['properties.address.formatted_address', 'properties.name',
                                     'lat', 'lng', 'id', 'duplicates']]

        # process labeled data
        train_test_data = self._process_labeled_data(labeled_data)

        # train model based on processed labeled data


        # evaluate model performance on hold out set


        # save trained model


    def _format_duplicates(self, duplicate_string):
        """
        Extracts the IDs of the duplicated POIs in a list format.

        :param duplicate_string: str
            Contains the duplicated IDs in string format.
        :return:
        duplicates: list
            Contains the duplicated IDs in list format.
        """
        duplicates = re.sub('[\[\]\']', '', duplicate_string).split(', ')
        if len(duplicates) == 1 and duplicates[0] == '':
            return []
        else:
            return duplicates

    def _process_labeled_data(self, labeled_data):
        """

        :param labeled_data:
        :return:
        """
        return labeled_data
