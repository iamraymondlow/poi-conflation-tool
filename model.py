import pandas as pd
import os
import json
import re
import numpy as np
import geopandas as gpd
import pyproj
import xgboost as xgb
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy.fuzz import token_set_ratio
from shapely.geometry import Point
from functools import partial
from shapely.ops import transform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, f1_score


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
        # load manually labeled data
        manual_data = pd.read_csv(os.path.join(os.path.dirname(__file__), config['labeled_data']))
        manual_data = manual_data.head(n=50)  # TODO remember to remove
        manual_data['duplicates'] = manual_data['duplicates'].apply(self._format_duplicates)
        manual_data = manual_data[['properties.address.formatted_address', 'properties.name',
                                   'lat', 'lng', 'id', 'duplicates']]
        manual_data = gpd.GeoDataFrame(manual_data,
                                       geometry=gpd.points_from_xy(manual_data['lng'],
                                                                   manual_data['lat']))

        # process manually labeled data
        train_test_data = self._process_manual_data(manual_data)
        train_test_data = pd.DataFrame(train_test_data, columns=['address_similarity', 'name_similarity', 'label'])

        # perform data sampling to balance class distribution
        train_datasets, test_data = self._perform_sampling(train_test_data)

        # train models
        gb_models, rf_models, xgboost_models = self._train(train_datasets)

        # evaluate model performance on hold out set
        y_pred_gb = self._predict(gb_models, test_data[['address_similarity', 'name_similarity']])
        y_pred_rf = self._predict(rf_models, test_data[['address_similarity', 'name_similarity']])
        y_pred_xgboost = self._predict(xgboost_models, test_data[['address_similarity', 'name_similarity']])
        self._evaluate(test_data['label'], y_pred_gb, 'Gradient Boosting')
        self._evaluate(test_data['label'], y_pred_rf, 'Random Forest')
        self._evaluate(test_data['label'], y_pred_xgboost, 'XGBoost')

        # save trained models locally
        # for i in range(len(gb_models)):
        #     dump(gb_models[i], 'gb_model{}.joblib'.format(i+1))

        return train_test_data

    def _evaluate(self, y_true, y_pred, algorithm):
        """
        Evaluates the model performance based on overall accuracy, balanced accuracy and macro-ave
        f1 score.

        :param y_true: Series
            Contains the ground truth labels for POI duplicates.
        :param y_pred: np.array
            Contains the model's inferred labels.
        :param algorithm: str
            Contains information about the algorithm under evaluation.
        """
        print(algorithm)
        print('Overall Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
        print('Balanced Accuracy: {}'.format(balanced_accuracy_score(y_true, y_pred)))
        print('Macro-average F1 Score: {}'.format(f1_score(y_true, y_pred, average='macro')))
        print(classification_report(y_true, y_pred, target_names=['Not Match', 'Match']))

    def _perform_sampling(self, data):
        """
        Performs bootstrapping and oversampling to rebalance the positive and negative class
        before creating subsets of the original dataset.

        :param data: Dataframe
            Contains the original dataset with imbalanced
        :return:
        train_datasets: list of Dataframe
            Contains subsets of the original dataset after bootstrapping and oversampling to
            rebalance the positive and negative classes.
        test_data: Dataframe
            Contains a randomly sampled hold out set of the original dataset for model evaluation.
        """
        train_datasets, test_data = None, None

        # train model based on processed labeled data
        # Extract the positive classes and negative classes
        # positive_data = labeled_data[labeled_data['label'] == 1].sample(frac=1).reset_index(drop=True)
        # negative_data = labeled_data[labeled_data['label'] == 0].sample(frac=1).reset_index(drop=True)
        #
        # # Oversample the positive class
        # positive_test = positive_data.iloc[:len(positive_data) // 4].reset_index(drop=True)
        # positive_train = positive_data.iloc[len(positive_data) // 4:].reset_index(drop=True)
        # print('Number of positive test data: {}'.format(len(positive_test)))
        # print('Number of positive train data before oversampling: {}'.format(len(positive_train)))
        # positive_train = pd.concat([positive_train, positive_train], ignore_index=True).sample(frac=1).reset_index(
        #     drop=True)
        # print('Number of positive train data after oversampling: {}'.format(len(positive_train)))
        #
        # # Undersampling the negative class
        # negative_test = negative_data.iloc[:len(negative_data) // 4].reset_index(drop=True)
        # negative_train = negative_data.iloc[len(negative_data) // 4:].reset_index(drop=True)
        # print('Number of negative test data: {}'.format(len(negative_test)))
        # print('Number of negative train data before undersampling: {}'.format(len(negative_train)))
        # negative_train_list = undersample(negative_train, len(positive_train))
        # print('Number of negative train data after undersampling: {}'.format(len(negative_train_list[0])))
        #
        # # Prepare test data
        # test_data = pd.concat([positive_test, negative_test], ignore_index=True).sample(frac=1).reset_index(drop=True)
        # X_test = test_data[['address_similarity', 'name_similarity']]
        # y_test = test_data['label']
        return train_datasets, test_data


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

    def _buffer_in_meters(self, lng, lat, radius):
        """
        Converts a latitude, longitude coordinate pair into a buffer with user-defined radius.s

        :param lng: float
            Contains the longitude information.
        :param lat: float
            Contains the latitude information.
        :param radius: float
            Contains the buffer radius in metres.
        :return:
        buffer_latlng: Polygon
            Contains the buffer.
        """
        proj_meters = pyproj.Proj(init='epsg:3414')  # EPSG for Singapore
        proj_latlng = pyproj.Proj(init='epsg:4326')

        project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
        project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)

        pt_meters = transform(project_to_meters, Point(lng, lat))

        buffer_meters = pt_meters.buffer(radius)
        buffer_latlng = transform(project_to_latlng, buffer_meters)
        return buffer_latlng

    def _label_data(self, manual_data, centroid_idx, address_matrix):
        """
        Generates the labeled data for the neighbouring POIs around a centroid POI.

        :param manual_data: GeoDataFrame
            Contains the manually labeled data and the ID information of their duplicates.
        :param centroid_idx: int
            Contains the index of the centroid POI.
        :param address_matrix: np.array
            Contains the address matrix after vectorising the address corpus using TFIDF.
        :return:
        """
        # identify neighbouring POIs
        buffer = self._buffer_in_meters(manual_data.loc[centroid_idx, 'lng'],
                                        manual_data.loc[centroid_idx, 'lat'],
                                        config['search_radius'])
        neighbour_pois = manual_data[manual_data.intersects(buffer)]
        neighbour_idx = list(neighbour_pois.index)

        # calculate address similarity score for neighbouring POIs
        centroid_address = address_matrix[centroid_idx, :]
        address_similarity = cosine_similarity(address_matrix[neighbour_idx, :], centroid_address).reshape(-1, 1)

        # calculate name similarity score for neighbouring POIs
        if pd.isnull(manual_data.loc[centroid_idx, 'properties.name']):
            return None
        name_similarity = np.array([token_set_ratio(manual_data.loc[centroid_idx, 'properties.name'], neighbour_name)
                                    for neighbour_name in
                                    manual_data.loc[neighbour_idx, 'properties.name'].tolist()]).reshape(-1, 1)

        # extract labels for neighbouring POIs
        labels = np.zeros((len(neighbour_idx), 1))
        for i in range(len(neighbour_idx)):
            if manual_data.loc[neighbour_idx[i], 'id'] in manual_data.loc[centroid_idx, 'duplicates']:
                labels[i, 0] = 1
            elif manual_data.loc[neighbour_idx[i], 'id'] == manual_data.loc[centroid_idx, 'id']:
                labels[i, 0] = 1
            else:
                pass

        return np.hstack((address_similarity, name_similarity, labels))

    def _process_manual_data(self, manual_data):
        """
        Processes the manually labeled data for model training and evaluation by identifying neighbouring POIs
        and labeling them as either duplicates or not duplicates.

        :param manual_data: GeoDataFrame
            Contains the manually labeled data and the ID information of their duplicates.
        :return:
        labeled_data: np.array
            Contains the labeled data ready for model training and evaluation.
        """
        address_corpus = manual_data['properties.address.formatted_address'].fillna('Singapore').tolist()
        address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        address_matrix = address_vectorizer.fit_transform(address_corpus)

        labeled_data = None
        for i in tqdm(range(len(manual_data))):
            temp_data = self._label_data(manual_data, i, address_matrix)

            if (temp_data is not None) and (labeled_data is not None):
                labeled_data = np.vstack((labeled_data, temp_data))
            elif temp_data is not None:
                labeled_data = temp_data
            else:
                pass

        return labeled_data

    def _undersample(self, data, num_positive_train):
        num_segment = (len(data) // num_positive_train) + 1
        return [
            data.iloc[(len(data) // num_segment) * i: (len(data) // num_segment) * (i + 1), :].reset_index(drop=True)
            for i in range(num_segment)]

    def _hyperparameter_tuning(self, train_data, algorithm):
        """
        Performs hyperparmeter tuning based on the training dataset.

        :param train_data: Dataframe
            Contains the training data.
        :param algorithm: str
            Indicates the algorithm used for model training.
        :return:
        grid_search: sklearn.model object
            Contains the trained model after hyperparameter tuning.
        """
        if algorithm == 'GB':
            parameters = {'n_estimators': np.arange(50, 210, 10),
                          'min_samples_split': np.arange(2, 6, 1),
                          'min_samples_leaf': np.arange(1, 6, 1),
                          'max_depth': np.arange(2, 6, 1)}
            model = GradientBoostingClassifier()

        elif algorithm == 'RF':
            parameters = {'n_estimators': np.arange(50, 210, 10),
                          'min_samples_split': np.arange(2, 6, 1),
                          'min_samples_leaf': np.arange(1, 6, 1)}
            model = RandomForestClassifier()

        elif algorithm == 'XGB':
            parameters = {'n_estimators': np.arange(50, 210, 10),
                          'learning_rate': [0.10, 0.20, 0.30],
                          'max_depth': [4, 6, 8, 10, 12, 15],
                          'min_child_weight': [1, 3, 5, 7],
                          'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                          'colsample_bytree': [0.3, 0.4, 0.5, 0.7]}
            model = xgb.XGBClassifier()

        else:
            raise ValueError('{} algorithm is not supported.'.format(algorithm))

        grid_search = GridSearchCV(model, parameters, scoring=['balanced_accuracy', 'f1_macro'],
                                   n_jobs=-1, refit='f1_macro')
        grid_search.fit(train_data[['address_similarity', 'name_similarity']],
                        train_data['label'])
        return grid_search

    def _train(self, train_datasets):
        """
        Performs model training based on different subsets of the training data and different algorithms.
        :param train_datasets: list of Dataframes
            Contains a list of training data after rebalancing the number of positive and
            negative classes.
        :return:
        gb_models, rf_models, xgboost_models: list of sklearn.models
            Contains the trained models based on different classification algorithms.
        """
        gb_models = []
        rf_models = []
        xgboost_models = []
        for train_data in train_datasets:
            gb_models.append(self._hyperparameter_tuning(train_data, 'GB'))
            rf_models.append(self._hyperparameter_tuning(train_data, 'RF'))
            xgboost_models.append(self._hyperparameter_tuning(train_data, 'XGB'))

        return gb_models, rf_models, xgboost_models

    def _predict(self, models, model_features):
        """
        Performs model prediction and combines the prediction made by all submodels by selecting
        the more likely classification.

        :param models: list of sklearn.models
            Contains the trained models.
        :param model_features: Dataframe
            Contain the model input features.
        :return:
        np.array
            Contains the most likely classification (duplicate or not) based on input features.
        """
        predict_prob = np.zeros((len(model_features), 2))
        for model in models:
            predict_prob += model.predict_proba(model_features)
        return np.argmax(predict_prob, axis=1)


if __name__ == '__main__':
    model = Model()
    data = model.train_model()
