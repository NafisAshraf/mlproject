import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation artifact path.
    """
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        """
        Initializes the data transformation configuration.
        """
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Builds and returns the scikit-learn preprocessing pipeline object.
        Combines numerical and categorical transformations into one ColumnTransformer.
        '''
        try:
            # Features to be imputed and scaled
            numerical_columns = ["writing_score", "reading_score"]
            # Features to be imputed, one-hot encoded, and scaled
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for handling numerical data: Impute missing with median, then scale
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            # Pipeline for handling categorical data: Impute missing, One-Hot encode, then scale
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combines both scaling pipelines into a single preprocessing object
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            # Standardized exception handling using CustomException
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        Applies pre-defined scaling and encoding to the train and test CSV data.
        Saves the resulting preprocessor object as a pickle file.
        Returns:
            Processeing train array, processing test array, and preprocessor file path.
        """
        try:
            # Reads the split CSV file paths provided after ingestion
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Get the transformation pipeline object
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Splitting feature inputs (X) and target outputs (y) for both datasets
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Fit preprocessor on training data, then transform both sets correctly
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features and target labels into solid arrays for training
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Persist the preprocessor state to disk as 'proprocessor.pkl'
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Standardized exception handling using CustomException
            raise CustomException(e,sys)