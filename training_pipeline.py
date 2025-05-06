import os
import json
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from abc import ABC, abstractmethod

# Import sklearn components
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Import algorithm-specific components

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    balanced_accuracy_score
)

# For clustering pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# For Time Series pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error

# Import custom utilities
import ml_utils
#import template_loader

# Initialize logger
logger = logging.getLogger(__name__)


#########################################
# Base Pipeline
#########################################

class BasePipeline(ABC):
    """
    Abstract base class for ML pipelines.
    
    This class defines the common structure and methods that all
    pipeline implementations should follow.
    """
    
    def __init__(
            self,
            data_path: Optional[str] = None,
            df: Optional[pd.DataFrame] = None,
            target: Optional[str] = None,
            config_path: Optional[str] = "config/config.yaml",
            output_dir: Optional[str] = None,
            model_id: Optional[str] = None
        ):
        """
        Initialize the base pipeline.
        
        Args:
            data_path: Path to the data file
            df: DataFrame (alternative to data_path)
            target: Name of the target column
            config_path: Path to configuration file
            output_dir: Directory for outputs
            model_id: Unique identifier for the model
        """
        # Initialize basic attributes
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_path = data_path
        self.df = df
        self.target = target
        self.model_id = model_id or f"model_{self.timestamp}"
        
        # Set up directories
        if output_dir:
            self.base_output_dir = output_dir
        else:
            self.base_output_dir = os.path.join("custom_ml_data", "training", "output", self.model_id)
        
        # Get version directory
        self.output_dir, self.version_num = ml_utils.get_next_version_dir(
            self.base_output_dir, self.model_id
        )
        self.version = f"v{self.version_num}"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load configuration
        try:
            self.config = ml_utils.load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}. Using defaults.")
            self.config = {}
        
        # Set up logging
        logging_config = self.config.get('common', {}).get('logging', {})
        self.logger = ml_utils.setup_logging(
            log_dir=self.output_dir,  # Use version folder directly, not a logs subfolder
            level=logging_config.get('level', 'INFO'),
            console_level=logging_config.get('console_level', 'INFO'),
            file_level=logging_config.get('file_level', 'DEBUG'),
            log_format=logging_config.get('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        logger.info(f"Logging initialized - Output directory: {self.output_dir}")
        # Initialize metadata
        self.metadata = ml_utils.initialize_metadata()
        self.metadata['parameters'].update({
            'data_path': self.data_path,
            'target': self.target,
            'model_id': self.model_id,
            'version': self.version
        })
        
        # Common parameters from config
        train_test_config = self.config.get('common', {}).get('train_test_split', {})
        self.test_size = train_test_config.get('test_size', 0.2)
        self.random_state = train_test_config.get('random_state', 42)
        
        # Initialize other common attributes
        self.problem_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = None
        self.best_model = None
        self.preprocessor = None
        
        logger.info(f"{self.__class__.__name__} initialized - ID: {self.model_id}, Version: {self.version}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file or use provided DataFrame.
        
        Returns:
            DataFrame: Loaded data
        """
        logger.info("Loading data...")
        
        if self.df is not None:
            logger.info("Using provided DataFrame")
            self.metadata['data']['source'] = 'provided_dataframe'
            self.metadata['data']['shape'] = self.df.shape
            return self.df
        
        if self.data_path is None:
            error_msg = "Either data_path or df must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        file_ext = os.path.splitext(self.data_path)[1].lower()
        self.metadata['data']['source'] = self.data_path
        self.metadata['data']['file_extension'] = file_ext
        
        try:
            start_time = datetime.now()
            
            if file_ext == '.csv':
                logger.info(f"Loading CSV file: {self.data_path}")
                self.df = pd.read_csv(self.data_path)
            elif file_ext in ['.xls', '.xlsx']:
                logger.info(f"Loading Excel file: {self.data_path}")
                self.df = pd.read_excel(self.data_path)
            elif file_ext == '.json':
                logger.info(f"Loading JSON file: {self.data_path}")
                self.df = pd.read_json(self.data_path)
            elif file_ext == '.parquet':
                logger.info(f"Loading Parquet file: {self.data_path}")
                self.df = pd.read_parquet(self.data_path)
            else:
                error_msg = f"Unsupported file extension: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data loaded in {load_time:.2f} seconds")
            
            # Update metadata
            self.metadata['data']['shape'] = self.df.shape
            self.metadata['data']['loading_time_seconds'] = load_time
            self.metadata['data']['columns'] = list(self.df.columns)
            self.metadata['data']['dtypes'] = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            
            # Calculate basic statistics for numeric columns
            numeric_columns = self.df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                # Basic descriptive statistics
                self.metadata['data']['numeric_stats'] = self.df[numeric_columns].describe().to_dict()
            
            # Track missing values
            missing_values = self.df.isnull().sum().to_dict()
            self.metadata['data']['missing_values'] = missing_values
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
        return self.df
    
    @abstractmethod
    def validate_data(self) -> bool:
        """
        Validate the input data and target.
        
        Each derived class must implement this.
        
        Returns:
            bool: True if validation passes
        """
        pass
    
    def detect_problem_type(self) -> str:
        """
        Detect the type of machine learning problem.
        
        Returns:
            str: Problem type ('regression', 'classification', etc.)
        """
        logger.info("Detecting problem type...")
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            self.problem_type = ml_utils.detect_problem_type(self.df, self.target, self.config)
            logger.info(f"Detected problem type: {self.problem_type}")
            
            # Save problem type and target column to metadata
            self.metadata['parameters']['problem_type'] = self.problem_type
            self.metadata['parameters']['target_column'] = self.target
            
            return self.problem_type
        except Exception as e:
            error_msg = f"Error detecting problem type: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def preprocess_data(self, custom_feature_engineering: Optional[Callable] = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess the data for machine learning.
        
        Args:
            custom_feature_engineering: Optional function for custom feature engineering
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Preprocessing data...")
        
        preprocessing_metadata = {}
        start_time = datetime.now()
        
        # Split into features and target
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        preprocessing_metadata['initial_shape'] = {'X': X.shape, 'y': y.shape}
        preprocessing_metadata['target_column'] = self.target
        
        # Apply custom feature engineering if provided
        if custom_feature_engineering:
            try:
                logger.info("Applying custom feature engineering...")
                X = custom_feature_engineering(X, self.config)
                logger.info(f"Features shape after engineering: {X.shape}")
                preprocessing_metadata['custom_features_applied'] = True
                preprocessing_metadata['features_after_engineering'] = X.shape
            except Exception as e:
                error_msg = f"Error in custom feature engineering: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.warning("Continuing with original features")
                preprocessing_metadata['custom_features_applied'] = False
                preprocessing_metadata['custom_features_error'] = str(e)
        
        # Identify timestamp columns
        timestamp_columns = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                timestamp_columns.append(col)
            # Try to convert string columns that might be timestamps
            elif X[col].dtype == 'object':
                try:
                    pd.to_datetime(X[col], errors='raise')
                    # Convert to actual datetime
                    X[col] = pd.to_datetime(X[col])
                    timestamp_columns.append(col)
                except:
                    pass
        
        logger.info(f"Identified {len(timestamp_columns)} timestamp columns")
        preprocessing_metadata['timestamp_columns'] = timestamp_columns
        
        # Get stratify parameter based on problem type
        stratify = None
        if self.problem_type == 'classification' and self.config.get('common', {}).get('train_test_split', {}).get('stratify', True):
            stratify = y
        
        # Split into train and test sets
        logger.info(f"Splitting data with test_size={self.test_size}, random_state={self.random_state}")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        logger.info(f"Train set size: {self.X_train.shape[0]}, Test set size: {self.X_test.shape[0]}")
        preprocessing_metadata['split_sizes'] = {
            'train': {'X': self.X_train.shape, 'y': self.y_train.shape},
            'test': {'X': self.X_test.shape, 'y': self.y_test.shape},
        }
        
        # Create preprocessing steps for different column types
        logger.info("Creating preprocessing pipeline...")
        
        # Get preprocessing config
        preproc_config = self.config.get('common', {}).get('preprocessing', {})
        outlier_config = preproc_config.get('outlier_detection', {})
        missing_values_config = preproc_config.get('missing_values', {})
        feature_eng_config = self.config.get('common', {}).get('feature_engineering', {})
        
        # Numeric transformer pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=missing_values_config.get('numeric_strategy', 'median'))),
            ('outlier', ml_utils.OutlierHandler(
                method=outlier_config.get('method', 'iqr'),
                threshold=outlier_config.get('threshold', 1.5),
                strategy=outlier_config.get('strategy', 'clip')
            ))
        ])
        
        # Add scaling if configured
        scaling_method = feature_eng_config.get('scaling', 'standard')
        if scaling_method != 'none':
            if scaling_method == 'standard':
                numeric_transformer.steps.append(('scaler', StandardScaler()))
            elif scaling_method == 'minmax':
                numeric_transformer.steps.append(('scaler', MinMaxScaler()))
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=missing_values_config.get('categorical_strategy', 'most_frequent'))),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Identify column types
        numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical_features = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        
        logger.info(f"Identified {len(numeric_features)} numeric features")
        logger.info(f"Identified {len(categorical_features)} categorical features")
        
        preprocessing_metadata['feature_types'] = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
        
        # Combine all preprocessing steps
        preprocessor_steps = []
        
        if numeric_features:
            preprocessor_steps.append(('numeric', numeric_transformer, numeric_features))
        
        if categorical_features:
            preprocessor_steps.append(('categorical', categorical_transformer, categorical_features))
        
        # Create the complete preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop'  # Drop any other columns
        )
        
        # Fit preprocessor on training data
        logger.info("Fitting preprocessor on training data...")
        self.preprocessor.fit(self.X_train)
        
        # Record preprocessing time
        preprocessing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        
        # Update metadata
        preprocessing_metadata['preprocessing_time_seconds'] = preprocessing_time
        preprocessing_metadata['preprocessor_steps'] = str(self.preprocessor)
        self.metadata['preprocessing'] = preprocessing_metadata
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    @abstractmethod
    def train_models(self) -> Dict[str, Any]:
        """
        Train machine learning models.
        
        Each derived class must implement this.
        
        Returns:
            dict: Trained models
        """
        pass
    
    @abstractmethod
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate trained models.
        
        Each derived class must implement this.
        
        Returns:
            DataFrame: Results with performance metrics
        """
        pass
    
    def save_model(self) -> str:
        """
        Save the best model and preprocessing pipeline.
        
        Returns:
            str: Path to saved model file
        """
        if self.best_model is None:
            error_msg = "No best model selected. Call evaluate_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Saving best model to {self.output_dir}")
        
        # Get model name
        best_model_name = self.metadata['best_model'].get('name', 'unknown')
        
        # Create a proper scikit-learn pipeline from components
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.best_model)
        ])
        
        # Use model_id as the filename
        model_filename = os.path.join(self.output_dir, f"{self.model_id}.pkl")
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(pipeline, f)
        
        logger.info(f"Best model ({best_model_name}) saved to {model_filename}")
        
        # Update metadata
        self.metadata['best_model']['filename'] = model_filename
        
        return model_filename
    
    def save_metadata(self) -> str:
        """
        Save metadata to file, with option to split large prediction data.
        
        Returns:
            str: Path to saved metadata file
        """
        metadata_copy = self.metadata.copy()
        predictions_samples = metadata_copy.get('best_model', {}).get('prediction_samples', [])
        
        if len(predictions_samples) > 1000:
            logger.info("Large number of prediction samples detected, saving to separate file")
            
            predictions_file = os.path.join(self.output_dir, f"{self.model_id}_prediction_samples.json")
            with open(predictions_file, 'w') as f:
                json.dump(predictions_samples, f, indent=2)
            
            metadata_copy['best_model']['prediction_samples'] = f"Saved separately to {predictions_file}"
            metadata_copy['best_model']['prediction_samples_file'] = predictions_file
        
        return ml_utils.save_metadata(
            metadata_copy,
            self.output_dir,
            filename="metadata.json"
        )

    def run_pipeline(self) -> Tuple[Any, pd.DataFrame]:
        """
        Run the complete pipeline.
        
        Returns:
            tuple: (best_model, evaluation_results)
        """
        logger.info(f"Starting ML pipeline - Problem type: {self.problem_type}")
        pipeline_start = datetime.now()
        
        try:
            # Load and validate data
            logger.info("\nLoading and validating data...")
            self.load_data()
            
            # Auto-detect problem type if not explicitly set
            if self.problem_type is None:
                self.detect_problem_type()
            
            self.metadata['parameters']['problem_type'] = self.problem_type


            # Validate data based on problem type
            self.validate_data()
            logger.info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Preprocess data
            logger.info("\nPreprocessing data...")
            self.preprocess_data()
            logger.info(f"Train set: {self.X_train.shape[0]} rows")
            logger.info(f"Test set: {self.X_test.shape[0]} rows")
            
            # Train models
            logger.info("\nTraining models...")
            self.train_models()
            logger.info(f"Trained {len(self.models)} models")
            
            # Evaluate models
            logger.info("\nEvaluating models...")
            self.evaluate_models()
            
            # Save the best model
            logger.info("\nSaving model...")
            self.save_model()
            
            # Calculate total runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"\nPipeline completed in {pipeline_runtime:.2f} seconds!")
            
            # Final metadata updates
            self.metadata['runtime_seconds'] = pipeline_runtime
            self.metadata['status'] = 'completed'
            self.save_metadata()
            
            return self.best_model, self.results
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            
            # Update metadata with error information
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.save_metadata()
            
            raise

#########################################
# Regression Pipeline
#########################################

class RegressionPipeline(BasePipeline):
    
    def __init__(
            self, 
            custom_models_function: Optional[Callable] = None,
            custom_features_function: Optional[Callable] = None,
            **kwargs
        ):
        """
        Initialize the regression pipeline.
        
        Args:
            custom_models_function: Optional function that creates custom models
            custom_features_function: Optional function for feature engineering
            **kwargs: Other arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.custom_models_function = custom_models_function
        self.custom_features_function = custom_features_function
        self.problem_type = 'regression'
        
        # Get regression-specific configuration
        self.model_config = ml_utils.get_model_config(self.config, 'regression')
        
        logger.info("Regression Pipeline initialized")
        if custom_models_function:
            logger.info("Custom models function provided")
        if custom_features_function:
            logger.info("Custom features function provided")
    
    def validate_data(self) -> bool:
        """
        Validate the data for regression tasks.
        
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating data for regression...")
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.target is None:
            error_msg = "Target column name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validation checks
        validation_results = {}
        
        # Check if target column exists
        if self.target not in self.df.columns:
            error_msg = f"Target column '{self.target}' not found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['target_exists'] = True
        
        # Check if dataset is empty
        if len(self.df) == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['dataset_not_empty'] = True
        
        # Check if target has valid numeric values
        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            error_msg = f"Target column '{self.target}' must contain numeric values for regression"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['target_is_numeric'] = True
        
        # Check for at least some valid values in target
        if self.df[self.target].isna().all():
            error_msg = f"Target column '{self.target}' contains only NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['target_has_values'] = True
        
        # Additional checks
        # Number of missing values in target
        missing_in_target = self.df[self.target].isna().sum()
        validation_results['missing_in_target'] = int(missing_in_target)
        logger.info(f"Missing values in target: {missing_in_target}")
        
        # Target distribution statistics
        target_stats = self.df[self.target].describe().to_dict()
        validation_results['target_stats'] = target_stats
        
        logger.info("Data validation complete")
        logger.info(f"Target column '{self.target}' summary: Min={target_stats['min']:.2f}, Max={target_stats['max']:.2f}, Mean={target_stats['mean']:.2f}")
        
        # Store validation results in metadata
        self.metadata['data']['validation'] = validation_results
        
        return True
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess the data for regression with optional custom feature engineering.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Call the parent class method with custom feature engineering if provided
        return super().preprocess_data(self.custom_features_function)
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train regression models, either custom or standard models.
        
        Returns:
            dict: Trained models
        """
        if self.preprocessor is None:
            error_msg = "Data not preprocessed. Call preprocess_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Training regression models...")
        training_metadata = {'models': {}}
        
        # Process training data once
        logger.info("Preprocessing training data...")
        X_train_processed = self.preprocessor.transform(self.X_train)
        
        # Check if we should use custom models
        use_custom_models = False
        custom_models = {}
        
        if not use_custom_models:
            # Get model configurations from config
            model_params = self.model_config.get('models', {}).get('parameters', {})
            enabled_models = self.model_config.get('models', {}).get('enabled', [])
            
            # Define all available regressors
            all_regressors = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=model_params.get('ridge', {}).get('alpha', 1.0),
                              random_state=self.random_state),
                'lasso': Lasso(alpha=model_params.get('lasso', {}).get('alpha', 0.1),
                              random_state=self.random_state),
                'elastic_net': ElasticNet(alpha=model_params.get('elastic_net', {}).get('alpha', 0.1),
                                        l1_ratio=model_params.get('elastic_net', {}).get('l1_ratio', 0.5),
                                        random_state=self.random_state),
                'decision_tree': DecisionTreeRegressor(
                    max_depth=model_params.get('decision_tree', {}).get('max_depth', None),
                    random_state=self.random_state
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=model_params.get('random_forest', {}).get('n_estimators', 100),
                    max_depth=model_params.get('random_forest', {}).get('max_depth', None),
                    random_state=self.random_state
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=model_params.get('gradient_boosting', {}).get('n_estimators', 100),
                    learning_rate=model_params.get('gradient_boosting', {}).get('learning_rate', 0.1),
                    random_state=self.random_state
                )
            }
            
            # Filter by enabled models if specified
            if enabled_models:
                regressors = {k: v for k, v in all_regressors.items() if k in enabled_models}
                logger.info(f"Training {len(regressors)} models: {', '.join(regressors.keys())}")
            else:
                regressors = all_regressors
                logger.info(f"Training all {len(regressors)} models")
        else:
            # Use the custom models
            regressors = custom_models
            training_metadata['custom_models_used'] = True
            training_metadata['custom_models'] = list(custom_models.keys())
        
        # Train all models
        self.models = {}
        
        for name, regressor in regressors.items():
            try:
                logger.info(f"Training {name}...")
                model_metadata = {
                    'model_type': str(type(regressor)),
                    'parameters': str(regressor.get_params())
                }
                
                # Time the training
                train_start = datetime.now()
                
                model = regressor.fit(X_train_processed, self.y_train)
                
                train_time = (datetime.now() - train_start).total_seconds()
                logger.info(f"Training {name} completed in {train_time:.2f} seconds")
                
                # Store the model
                self.models[name] = model
                
                # Update metadata
                model_metadata['training_time_seconds'] = train_time
                model_metadata['trained_successfully'] = True
                
                training_metadata['models'][name] = model_metadata
                
            except Exception as e:
                error_msg = f"Failed to train {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                training_metadata['models'][name] = {
                    'trained_successfully': False,
                    'error': str(e)
                }
        
        logger.info(f"Successfully trained {len(self.models)} models")
        
        # Update metadata
        self.metadata['models'] = training_metadata
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate regression models and store primary metric for each model.
        
        Returns:
            pd.DataFrame: Results with performance metrics
        """
        if not self.models:
            error_msg = "No trained models. Call train_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Evaluating regression models...")
        evaluation_metadata = {'models': {}}
        
        # Get evaluation configuration
        eval_metrics = self.model_config.get('evaluation', {}).get('metrics', ['r2', 'rmse', 'mae'])
        primary_metric = self.model_config.get('evaluation', {}).get('primary_metric', 'r2')
        
        # Transform test data
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        results = []
        
        # Store primary metric for each model
        all_models_primary_metric = {}
        

        for name, model in self.models.items():
            try:
                logger.info(f"Evaluating {name}...")
                model_eval_metadata = {}
                
                # Make predictions
                y_train_pred = model.predict(self.preprocessor.transform(self.X_train))
                y_test_pred = model.predict(X_test_processed)
                
                # Calculate metrics
                metrics = {
                    'model': name,
                    'train_r2': r2_score(self.y_train, y_train_pred),
                    'test_r2': r2_score(self.y_test, y_test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                    'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                    'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                    'train_mape': mean_absolute_percentage_error(self.y_train, y_train_pred),
                    'test_mape': mean_absolute_percentage_error(self.y_test, y_test_pred)

                }
                
                # Save primary metric value for this model
                metric_col = f'test_{primary_metric}'
                primary_metric_value = metrics[metric_col]
                all_models_primary_metric[name] = float(primary_metric_value)
                
                results.append(metrics)
                
                # Update metadata
                model_eval_metadata['metrics'] = {
                    'train_r2': float(metrics['train_r2']),
                    'test_r2': float(metrics['test_r2']),
                    'train_rmse': float(metrics['train_rmse']),
                    'test_rmse': float(metrics['test_rmse']),
                    'train_mae': float(metrics['train_mae']),
                    'test_mae': float(metrics['test_mae']),
                    'train_mape': float(metrics['train_mape']),
                    'test_mape': float(metrics['test_mape'])
                }
                
                # Log results
                logger.info(f"  {name}:")
                logger.info(f"    Train R²: {metrics['train_r2']:.4f}, Test R²: {metrics['test_r2']:.4f}")
                logger.info(f"    Train RMSE: {metrics['train_rmse']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}")
                logger.info(f"    Train MAE: {metrics['train_mae']:.4f}, Test MAE: {metrics['test_mae']:.4f}")
                logger.info(f"    Train MAPE: {metrics['train_mape']:.4f}, Test MAPE: {metrics['test_mape']:.4f}")

                
                evaluation_metadata['models'][name] = model_eval_metadata
                
            except Exception as e:
                error_msg = f"Failed to evaluate {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                evaluation_metadata['models'][name] = {
                    'evaluation_error': str(e)
                }
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        evaluation_metadata['results_summary'] = self.results.to_dict()
        
        # Store all models primary metric in metadata
        evaluation_metadata['primary_metric'] = primary_metric
        evaluation_metadata['all_models_primary_metric'] = all_models_primary_metric
        
        # Identify best model based on primary metric
        if not self.results.empty:
            metric_col = f'test_{primary_metric}'
            
            if primary_metric in ['rmse', 'mae']:  # These are error metrics, lower is better
                best_idx = self.results[metric_col].idxmin()
            else:  # For R², higher is better
                best_idx = self.results[metric_col].idxmax()
                
            best_model_name = self.results.loc[best_idx, 'model']
            best_metric_value = self.results.loc[best_idx, metric_col]
            
            self.best_model = self.models[best_model_name]
            logger.info(f"\nBest model: {best_model_name} (Test {primary_metric.upper()} = {best_metric_value:.4f})")
            
            # Get predictions from the best model
            best_test_predictions = self.best_model.predict(X_test_processed)

            # Store predictions in metadata
            self._store_best_model_predictions(best_model_name, best_test_predictions)

            # Update metadata
            self.metadata['best_model']['name'] = best_model_name
            self.metadata['best_model'][f'test_{primary_metric}'] = float(best_metric_value)
            self.metadata['best_model']['metrics'] = {
                'test_r2': float(self.results.loc[best_idx, 'test_r2']),
                'test_rmse': float(self.results.loc[best_idx, 'test_rmse']),
                'test_mae': float(self.results.loc[best_idx, 'test_mae']),
                'test_mape': float(self.results.loc[best_idx, 'test_mape'])

            }
        
        # Update overall metadata
        self.metadata['evaluation'] = evaluation_metadata
        
        return self.results

    def _store_best_model_predictions(self, model_name, predictions, max_predictions=1000):
        """
        Store predictions from the best model in the metadata.
        
        Args:
            model_name: Name of the best model
            predictions: Array of predictions on the test set
            max_predictions: Maximum number of predictions to store in metadata
        """
        logger.info(f"Storing predictions for best model: {model_name}")
        
        # Create a DataFrame with actual and predicted values
        prediction_df = pd.DataFrame({
            'actual': self.y_test.values,
            'predicted': predictions
        })
        
        # If there's an index in the original test data, try to preserve it
        if hasattr(self.y_test, 'index') and self.y_test.index is not None:
            prediction_df.index = self.y_test.index
        
        # Sample predictions if too many
        if len(prediction_df) > max_predictions:
            sampled_predictions = prediction_df.sample(n=max_predictions, random_state=42)
            logger.info(f"Storing {len(sampled_predictions)} predictions in metadata (sampled from {len(prediction_df)} total)")
        else:
            # Store all predictions if under the limit
            sampled_predictions = prediction_df
            logger.info(f"Storing all {len(prediction_df)} predictions in metadata")
        
        # Convert to list for storing in metadata
        predictions_list = []
        for idx, row in sampled_predictions.iterrows():
            predictions_list.append({
                'actual': float(row['actual']),
                'predicted': float(row['predicted'])
            })
        
        # Add to metadata
        self.metadata['best_model']['prediction_samples'] = predictions_list

#########################################
# Classification Pipeline
#########################################

class ClassificationPipeline(BasePipeline):
  
    def __init__(
            self, 
            custom_models_function: Optional[Callable] = None,
            custom_features_function: Optional[Callable] = None,
            balance_method: Optional[str] = None,
            multi_class: str = 'auto',
            eval_metric: str = 'accuracy',
            **kwargs
        ):
        """
        Initialize the classification pipeline.
        
        Args:
            custom_models_function: Optional function that creates custom models
            custom_features_function: Optional function for feature engineering
            balance_method: Method to balance classes (None, 'smote', 'class_weight')
            multi_class: Strategy for multi-class problems ('auto', 'ovr', 'multinomial')
            eval_metric: Primary evaluation metric
            **kwargs: Other arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.custom_models_function = custom_models_function
        self.custom_features_function = custom_features_function
        self.problem_type = 'classification'
        self.balance_method = balance_method
        self.multi_class = multi_class
        self.eval_metric = eval_metric
        
        # Get classification-specific configuration
        self.model_config = ml_utils.get_model_config(self.config, 'classification')
        
        # Initialize classification-specific attributes
        self.le = None  # Label encoder for target
        self.class_names = None  # Original class names
        self.is_binary = None  # Flag for binary classification
        self.class_distribution = None  # Store class distribution
        
        logger.info("Classification Pipeline initialized")
        if custom_models_function:
            logger.info("Custom models function provided")
        if custom_features_function:
            logger.info("Custom features function provided")
    
    def validate_data(self) -> bool:
        """
        Validate the data for classification tasks.
        
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating data for classification...")
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.target is None:
            error_msg = "Target column name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validation checks
        validation_results = {}
        
        # Check if target column exists
        if self.target not in self.df.columns:
            error_msg = f"Target column '{self.target}' not found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['target_exists'] = True
        
        # Check if dataset is empty
        if len(self.df) == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['dataset_not_empty'] = True
        
        # Check target class distribution
        self.class_distribution = self.df[self.target].value_counts()
        validation_results['class_distribution'] = self.class_distribution.to_dict()
        
        if len(self.class_distribution) < 2:
            error_msg = f"Target column '{self.target}' must have at least 2 classes"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['has_multiple_classes'] = True
        
        # Identify binary or multiclass problem
        self.is_binary = len(self.class_distribution) == 2
        self.problem_subtype = 'binary' if self.is_binary else 'multiclass'
        validation_results['problem_subtype'] = self.problem_subtype
        
        # Check for class imbalance
        min_class_count = self.class_distribution.min()
        max_class_count = self.class_distribution.max()
        imbalance_ratio = max_class_count / min_class_count
        validation_results['imbalance_ratio'] = float(imbalance_ratio)
        
        if imbalance_ratio > 3:
            logger.warning(f"Class imbalance detected (ratio {imbalance_ratio:.2f})")
            if self.balance_method is None:
                logger.warning("Consider setting balance_method to handle class imbalance")
        
        logger.info(f"Classification problem subtype: {self.problem_subtype}")
        logger.info("Class distribution:")
        for cls, count in self.class_distribution.items():
            logger.info(f"  {cls}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Store validation results in metadata
        self.metadata['data']['validation'] = validation_results
        
        return True
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess the data for classification with optional custom feature engineering.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Call the parent method first to get basic preprocessing
        X_train, X_test, y_train, y_test = super().preprocess_data(self.custom_features_function)
        
        # Encode target if it's categorical
        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            logger.info("Encoding categorical target variable")
            self.le = LabelEncoder()
            self.y_train = self.le.fit_transform(y_train)
            self.y_test = self.le.transform(y_test)
            self.class_names = list(self.le.classes_)
            class_mapping = dict(zip(self.class_names, range(len(self.class_names))))
            logger.info(f"Target encoded: {class_mapping}")
            self.metadata['preprocessing']['class_encoding'] = class_mapping
        else:
            logger.info("Target is already numeric, no encoding needed")
            self.class_names = [str(cls) for cls in sorted(self.df[self.target].unique())]
            self.metadata['preprocessing']['class_names'] = self.class_names
            
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _get_primary_metric(self):
        """Get the primary metric function based on user selection."""
        metric_name = self.eval_metric
        logger.debug(f"Setting primary metric: {metric_name}")
        
        if metric_name == 'accuracy':
            return accuracy_score
        elif metric_name == 'balanced_accuracy':
            return balanced_accuracy_score
        elif metric_name == 'precision':
            return lambda y_true, y_pred: precision_score(
                y_true, y_pred, average='binary' if self.is_binary else 'macro'
            )
        elif metric_name == 'recall':
            return lambda y_true, y_pred: recall_score(
                y_true, y_pred, average='binary' if self.is_binary else 'macro'
            )
        elif metric_name == 'f1':
            return lambda y_true, y_pred: f1_score(
                y_true, y_pred, average='binary' if self.is_binary else 'macro'
            )
        elif metric_name == 'auc' and self.is_binary:
            return roc_auc_score
        else:
            # Default to accuracy if invalid metric for problem type
            logger.warning(f"Metric '{metric_name}' not suitable for {self.problem_subtype} classification, using accuracy")
            return accuracy_score
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train classification models, either custom or standard models.
        
        Returns:
            dict: Trained models
        """
        if self.preprocessor is None:
            error_msg = "Data not preprocessed. Call preprocess_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Training classification models...")
        training_metadata = {'models': {}}
        
        # Process training data once
        logger.info("Preprocessing training data...")
        X_train_processed = self.preprocessor.transform(self.X_train)
        
        # Define class weights if needed
        class_weights = None
        if self.balance_method == 'class_weight':
            class_counts = np.bincount(self.y_train) if self.is_binary else np.unique(self.y_train, return_counts=True)[1]
            total = len(self.y_train)
            class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
            logger.info(f"Using class weights: {class_weights}")
            training_metadata['class_weights'] = class_weights
        
        # Apply SMOTE if requested
        if self.balance_method == 'smote':
            logger.info("Applying SMOTE to balance classes...")
            try:
                from imblearn.over_sampling import SMOTE
                smote_start = datetime.now()
                smote = SMOTE(random_state=self.random_state)
                X_train_processed, y_train_resampled = smote.fit_resample(X_train_processed, self.y_train)
                smote_time = (datetime.now() - smote_start).total_seconds()
                
                # Log new class distribution
                unique, counts = np.unique(y_train_resampled, return_counts=True)
                class_dist_after_smote = {}
                logger.info("Class distribution after SMOTE:")
                for cls, count in zip(unique, counts):
                    cls_name = self.class_names[cls] if self.le is not None else cls
                    class_dist_after_smote[str(cls_name)] = int(count)
                    logger.info(f"  {cls_name}: {count} ({count/len(y_train_resampled)*100:.1f}%)")
                
                training_metadata['smote'] = {
                    'applied': True,
                    'time_seconds': smote_time,
                    'class_distribution_after': class_dist_after_smote,
                    'total_samples_after': int(len(y_train_resampled))
                }
            except Exception as e:
                error_msg = f"Error applying SMOTE: {str(e)}. Falling back to original data."
                logger.error(error_msg, exc_info=True)
                y_train_resampled = self.y_train
                training_metadata['smote'] = {
                    'applied': False,
                    'error': str(e)
                }
        else:
            y_train_resampled = self.y_train
            training_metadata['smote'] = {'applied': False}
        
        # Check if we should use custom models
        use_custom_models = False
        custom_models = {}
        
        if not use_custom_models:
            # Get model configurations from config
            model_params = self.model_config.get('models', {}).get('parameters', {})
            enabled_models = self.model_config.get('models', {}).get('enabled', [])
            
            # Define all available classifiers
            all_classifiers = {
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    class_weight='balanced' if self.balance_method == 'class_weight' else None,
                    multi_class=self.multi_class if not self.is_binary else 'ovr',
                    max_iter=1000
                ),
                'decision_tree': DecisionTreeClassifier(
                    random_state=self.random_state,
                    class_weight='balanced' if self.balance_method == 'class_weight' else None
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=model_params.get('random_forest', {}).get('n_estimators', 100),
                    max_depth=model_params.get('random_forest', {}).get('max_depth', None),
                    class_weight='balanced' if self.balance_method == 'class_weight' else None,
                    random_state=self.random_state
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=model_params.get('gradient_boosting', {}).get('n_estimators', 100),
                    learning_rate=model_params.get('gradient_boosting', {}).get('learning_rate', 0.1),
                    random_state=self.random_state
                ),
                'knn': KNeighborsClassifier(
                    n_neighbors=model_params.get('knn', {}).get('n_neighbors', 5)
                ),
                'naive_bayes': GaussianNB()
            }
            
            # Add more complex models for smaller datasets
            if len(self.X_train) < 10000:
                logger.info("Dataset size < 10000 rows, adding SVC and MLP models")
                all_classifiers['svc'] = SVC(
                    kernel='rbf',
                    gamma='scale',
                    probability=True,
                    class_weight='balanced' if self.balance_method == 'class_weight' else None,
                    random_state=self.random_state
                )
                all_classifiers['mlp'] = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_state
                )
            
            # Filter by enabled models if specified
            if enabled_models:
                classifiers = {k: v for k, v in all_classifiers.items() if k in enabled_models}
                logger.info(f"Training {len(classifiers)} models: {', '.join(classifiers.keys())}")
            else:
                classifiers = all_classifiers
                logger.info(f"Training all {len(classifiers)} models")
        else:
            # Use the custom models
            classifiers = custom_models
            training_metadata['custom_models_used'] = True
            training_metadata['custom_models'] = list(custom_models.keys())
            
        # Train all models
        self.models = {}
        
        for name, classifier in classifiers.items():
            try:
                logger.info(f"Training {name}...")
                model_metadata = {
                    'model_type': str(type(classifier)),
                    'parameters': str(classifier.get_params())
                }
                
                # Time the training
                train_start = datetime.now()
                
                model = classifier.fit(X_train_processed, y_train_resampled)
                
                train_time = (datetime.now() - train_start).total_seconds()
                logger.info(f"Training {name} completed in {train_time:.2f} seconds")
                
                # Store the model
                self.models[name] = model
                
                # Update metadata
                model_metadata['training_time_seconds'] = train_time
                model_metadata['trained_successfully'] = True
                
                training_metadata['models'][name] = model_metadata
                
            except Exception as e:
                error_msg = f"Failed to train {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                training_metadata['models'][name] = {
                    'trained_successfully': False,
                    'error': str(e)
                }
        
        logger.info(f"Successfully trained {len(self.models)} models")
        
        # Update metadata
        self.metadata['models'] = training_metadata
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate classification models.
        
        Returns:
            pd.DataFrame: Results with performance metrics
        """
        if not self.models:
            error_msg = "No trained models. Call train_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get primary metric function
        primary_metric = self._get_primary_metric()
        primary_metric_name = self.eval_metric
        
        logger.info("Evaluating classification models...")
        evaluation_metadata = {'models': {}}
        
        # Transform test data
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        results = []
        
        for name, model in self.models.items():
            try:
                logger.info(f"Evaluating {name}...")
                model_eval_metadata = {}
                
                # Make predictions
                y_train_pred = model.predict(self.preprocessor.transform(self.X_train))
                y_test_pred = model.predict(X_test_processed)
                
                # Calculate accuracy
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                
                # Calculate primary metric
                train_primary = primary_metric(self.y_train, y_train_pred)
                test_primary = primary_metric(self.y_test, y_test_pred)
                
                # Store results
                result = {
                    'model': name,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    f'train_{primary_metric_name}': train_primary,
                    f'test_{primary_metric_name}': test_primary
                }
                
                # Add AUC for binary problems if classifier supports predict_proba
                if self.is_binary and hasattr(model, 'predict_proba'):
                    try:
                        y_test_prob = model.predict_proba(X_test_processed)[:, 1]
                        auc = roc_auc_score(self.y_test, y_test_prob)
                        result['test_auc'] = auc
                        model_eval_metadata['auc'] = float(auc)
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC for {name}: {str(e)}")
                        result['test_auc'] = np.nan
                
                results.append(result)
                
                # Calculate confusion matrix
                cm = confusion_matrix(self.y_test, y_test_pred)
                model_eval_metadata['confusion_matrix'] = cm.tolist()
                
                # Calculate detailed classification report
                try:
                    target_names = [str(name) for name in self.class_names]
                    cls_report = classification_report(self.y_test, y_test_pred, 
                                                      target_names=target_names, 
                                                      output_dict=True)
                    model_eval_metadata['classification_report'] = cls_report
                except Exception as e:
                    logger.warning(f"Could not generate classification report for {name}: {str(e)}")
                
                # Update metadata with metrics
                model_eval_metadata['metrics'] = {
                    'train_accuracy': float(train_accuracy),
                    'test_accuracy': float(test_accuracy),
                    f'train_{primary_metric_name}': float(train_primary),
                    f'test_{primary_metric_name}': float(test_primary)
                }
                
                evaluation_metadata['models'][name] = model_eval_metadata
                
                # Log results
                logger.info(f"  {name}:")
                logger.info(f"    Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
                
                if self.is_binary:
                    precision = precision_score(self.y_test, y_test_pred)
                    recall = recall_score(self.y_test, y_test_pred)
                    f1 = f1_score(self.y_test, y_test_pred)
                    logger.info(f"    Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}")
                else:
                    precision = precision_score(self.y_test, y_test_pred, average='macro')
                    recall = recall_score(self.y_test, y_test_pred, average='macro')
                    f1 = f1_score(self.y_test, y_test_pred, average='macro')
                    logger.info(f"    Test Precision (macro): {precision:.4f}, Test Recall (macro): {recall:.4f}, Test F1 (macro): {f1:.4f}")
                
                # Log confusion matrix
                logger.info("    Confusion Matrix:")
                logger.info(f"{cm}")
                
            except Exception as e:
                error_msg = f"Failed to evaluate {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                evaluation_metadata['models'][name] = {
                    'evaluation_error': str(e)
                }
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        evaluation_metadata['results_summary'] = self.results.to_dict()
        
        # Identify best model based on primary metric
        if not self.results.empty:
            # Adjust metric name for multiclass case
            if not self.is_binary and primary_metric_name in ['precision', 'recall', 'f1']:
                metric_col = f'test_{primary_metric_name}_macro'
            else:
                metric_col = f'test_{primary_metric_name}'
            
            # Find best model
            if metric_col in self.results.columns:
                best_idx = self.results[metric_col].idxmax()
                best_model_name = self.results.loc[best_idx, 'model']
                best_metric_value = self.results.loc[best_idx, metric_col]
                
                self.best_model = self.models[best_model_name]
                logger.info(f"\nBest model: {best_model_name} (Test {primary_metric_name} = {best_metric_value:.4f})")
                
                # Get predictions from the best model
                best_test_predictions = self.best_model.predict(X_test_processed)
                self._store_best_model_predictions(best_model_name, best_test_predictions)

                self.best_model = self.models[best_model_name]
                logger.info(f"\nBest model: {best_model_name} (Test {primary_metric_name} = {best_metric_value:.4f})")

                best_test_predictions = self.best_model.predict(X_test_processed)
                self._store_best_model_predictions(best_model_name, best_test_predictions)


                # Update metadata
                self.metadata['best_model']['name'] = best_model_name
                self.metadata['best_model'][metric_col] = float(best_metric_value)
                self.metadata['best_model']['metrics'] = {
                    'test_accuracy': float(self.results.loc[best_idx, 'test_accuracy'])
                }
                
                # Add more metrics based on problem type
                if self.is_binary and 'test_auc' in self.results.columns:
                    auc_value = self.results.loc[best_idx, 'test_auc']
                    if not pd.isna(auc_value):
                        self.metadata['best_model']['metrics']['test_auc'] = float(auc_value)
            else:
                logger.warning(f"Primary metric '{metric_col}' not found in results. Using accuracy instead.")
                best_idx = self.results['test_accuracy'].idxmax()
                best_model_name = self.results.loc[best_idx, 'model']
                self.best_model = self.models[best_model_name]
                
                # Update metadata
                self.metadata['best_model']['name'] = best_model_name
                self.metadata['best_model']['test_accuracy'] = float(self.results.loc[best_idx, 'test_accuracy'])
        
        # Update overall metadata
        self.metadata['evaluation'] = evaluation_metadata
        
        return self.results

    def _store_best_model_predictions(self, model_name, predictions, max_predictions=1000):
        """
        Store predictions from the best classification model.
        
        Args:
            model_name: Name of the best model
            predictions: Array of predictions on the test set
            max_predictions: Maximum number of predictions to store
        """
        logger.info(f"Storing predictions for best model: {model_name}")
        
        # Get the original class names
        if hasattr(self, 'le') and self.le is not None:
            # Get encoded predictions and actual values
            y_test_encoded = self.y_test
            predictions_encoded = predictions
            
            # Decode them back to original labels
            try:
                y_test_decoded = self.le.inverse_transform(y_test_encoded)
                predictions_decoded = self.le.inverse_transform(predictions_encoded)
                
                # Create a DataFrame with actual and predicted values (using decoded values)
                prediction_df = pd.DataFrame({
                    'actual': y_test_decoded,
                    'predicted': predictions_decoded
                })
                
                # Log to confirm correct decoding
                logger.info(f"Decoded classes - first 5 samples:")
                for i in range(min(5, len(y_test_decoded))):
                    logger.info(f"  Actual: {y_test_decoded[i]}, Predicted: {predictions_decoded[i]}")
                    
            except Exception as e:
                logger.error(f"Failed to decode class labels: {str(e)}")
                # Fallback to encoded values
                prediction_df = pd.DataFrame({
                    'actual': y_test_encoded,
                    'predicted': predictions_encoded
                })
        else:
            # No encoding was done, use original values
            prediction_df = pd.DataFrame({
                'actual': self.y_test,
                'predicted': predictions
            })
        
        # Add correct/incorrect column
        prediction_df['correct'] = prediction_df['actual'] == prediction_df['predicted']
        
        # Define incorrect_predictions AFTER creating the 'correct' column
        incorrect_predictions = prediction_df[~prediction_df['correct']]
        
        # If there's an index in the original test data, try to preserve it
        if hasattr(self.y_test, 'index') and self.y_test.index is not None:
            prediction_df.index = self.y_test.index
        
        # Get class labels if available
        class_labels = None
        if hasattr(self.preprocessor, 'named_transformers_') and hasattr(self.preprocessor.named_transformers_, 'cat'):
            if hasattr(self.preprocessor.named_transformers_.cat, 'named_steps'):
                if hasattr(self.preprocessor.named_transformers_.cat.named_steps, 'encoder'):
                    encoder = self.preprocessor.named_transformers_.cat.named_steps.encoder
                    if hasattr(encoder, 'categories_'):
                        class_labels = encoder.categories_
        
        # Save confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, predictions)
        
        # Convert confusion matrix to a list for JSON serialization
        cm_list = cm.tolist()
        
        # Calculate counts and percentages of correct/incorrect predictions
        correct_count = prediction_df['correct'].sum()
        total_count = len(prediction_df)
        accuracy = correct_count / total_count
        
        prediction_stats = {
            'correct_count': int(correct_count),
            'incorrect_count': int(total_count - correct_count),
            'total_count': int(total_count),
            'accuracy': float(accuracy),
            'confusion_matrix': cm_list
        }
        
        # Add class distribution if available
        actual_class_counts = pd.Series(self.y_test).value_counts().to_dict()
        predicted_class_counts = pd.Series(predictions).value_counts().to_dict()
        
        # Convert to string keys for JSON serialization
        prediction_stats['actual_class_distribution'] = {str(k): int(v) for k, v in actual_class_counts.items()}
        prediction_stats['predicted_class_distribution'] = {str(k): int(v) for k, v in predicted_class_counts.items()}
        
        # Store class labels if available
        if class_labels is not None:
            prediction_stats['class_labels'] = class_labels
        
        # Store prediction statistics in metadata
        self.metadata['best_model']['prediction_stats'] = prediction_stats
        
        # Store a sample of predictions in metadata
        if len(prediction_df) > max_predictions:
            # For classification, stratify by correct/incorrect and class
            # We already defined incorrect_predictions above, so no need to redefine
            correct_predictions = prediction_df[prediction_df['correct']]
            
            # Take more incorrect examples as they're more interesting
            incorrect_sample_size = min(max_predictions // 2, len(incorrect_predictions))
            correct_sample_size = max_predictions - incorrect_sample_size
            
            # Sample from each group
            sampled_incorrect = incorrect_predictions.sample(
                n=incorrect_sample_size, 
                random_state=42
            ) if len(incorrect_predictions) > 0 else pd.DataFrame()
            
            sampled_correct = correct_predictions.sample(
                n=correct_sample_size, 
                random_state=42
            ) if len(correct_predictions) > 0 else pd.DataFrame()
            
            # Combine samples
            sampled_predictions = pd.concat([sampled_incorrect, sampled_correct])
            
            logger.info(f"Storing {len(sampled_predictions)} predictions in metadata (sampled from {len(prediction_df)} total)")
        else:
            # Store all if under the limit
            sampled_predictions = prediction_df
            logger.info(f"Storing all {len(prediction_df)} predictions in metadata")
        
        # Convert to list for storing in metadata
        predictions_list = []
        for idx, row in sampled_predictions.iterrows():
            predictions_list.append({
                'actual': str(row['actual']),
                'predicted': str(row['predicted'])
            })
        
        # Add to metadata
        self.metadata['best_model']['prediction_samples'] = predictions_list
        
        # Add all incorrect predictions (up to a limit)
        max_incorrect = min(20, len(incorrect_predictions))
        if max_incorrect > 0:
            incorrect_list = []
            for idx, row in incorrect_predictions.head(max_incorrect).iterrows():
                incorrect_list.append({
                    'actual': str(row['actual']),
                    'predicted': str(row['predicted'])
                })
            
            self.metadata['best_model']['incorrect_predictions'] = incorrect_list

#########################################
# Clustering Pipeline
#########################################

class ClusteringPipeline(BasePipeline):
    
    def __init__(
            self,
            data_path: Optional[str] = None,
            df: Optional[pd.DataFrame] = None,
            n_clusters: Optional[int] = None,
            config_path: Optional[str] = "config/config.yaml",
            output_dir: Optional[str] = None,
            model_id: Optional[str] = None,
            cluster_range: Optional[Tuple[int, int]] = (2, 15),
            dim_reduction: str = 'pca',
            n_components: int = 2
        ):
        """
        Initialize the clustering pipeline.
        
        Args:
            data_path: Path to the data file
            df: DataFrame (alternative to data_path)
            n_clusters: Number of clusters (if None, will be estimated)
            config_path: Path to configuration file
            output_dir: Directory for outputs
            model_id: Unique identifier for the model
            cluster_range: Range of clusters to try for optimal cluster selection
            dim_reduction: Dimensionality reduction method ('pca', 'tsne', 'none')
            n_components: Number of components for dimensionality reduction
        """
        # Call the parent class constructor without a target variable
        super().__init__(data_path=data_path, df=df, target=None, 
                          config_path=config_path, output_dir=output_dir, model_id=model_id)
        
        self.problem_type = 'clustering'
        self.n_clusters = n_clusters
        self.cluster_range = cluster_range
        self.dim_reduction = dim_reduction
        self.n_components = n_components
        
        # Get clustering-specific configuration
        self.model_config = ml_utils.get_model_config(self.config, 'clustering')
        
        # Initialize clustering-specific attributes
        self.X_scaled = None
        self.X_reduced = None
        self.optimal_clusters = n_clusters
        self.feature_importances = None
        
        logger.info("Clustering Pipeline initialized")
        if n_clusters is not None:
            logger.info(f"Using specified number of clusters: {n_clusters}")
        else:
            logger.info(f"Will estimate optimal number of clusters in range: {cluster_range}")
    
    def validate_data(self) -> bool:
        """
        Validate the data for clustering tasks.
        
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating data for clustering...")
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validation checks
        validation_results = {}
        
        # Check if dataset is empty
        if len(self.df) == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['dataset_not_empty'] = True
        validation_results['rows'] = len(self.df)
        validation_results['columns'] = len(self.df.columns)
        
        # Analyze data types
        numeric_columns = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        validation_results['column_types'] = {
            'numeric': len(numeric_columns),
            'categorical': len(categorical_columns),
            'datetime': len(datetime_columns)
        }
        
        if not numeric_columns:
            logger.warning("No numeric columns found. Clustering typically works best with numeric data.")
            validation_results['has_numeric_columns'] = False
        else:
            validation_results['has_numeric_columns'] = True
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if not columns_with_missing.empty:
            missing_data = {}
            for col, count in columns_with_missing.items():
                percent = (count / len(self.df)) * 100
                logger.info(f"  {col}: {count} missing values ({percent:.2f}%)")
                missing_data[col] = {
                    'count': int(count),
                    'percent': float(percent)
                }
            validation_results['columns_with_missing'] = missing_data
        else:
            validation_results['columns_with_missing'] = {}
        
        # Store validation results in metadata
        self.metadata['data']['validation'] = validation_results
        
        logger.info(f"Data validated: {len(self.df.columns)} feature columns available for clustering")
        
        return True
    
    def preprocess_data(self) -> np.ndarray:
        """
        Preprocess the data for clustering.
        
        Returns:
            np.ndarray: Preprocessed data ready for clustering
        """
        logger.info("Preprocessing data...")
        
        preprocessing_metadata = {}
        start_time = datetime.now()
        
        # Create preprocessing steps for different column types
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier', ml_utils.OutlierHandler(method='iqr', threshold=1.5, strategy='clip'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Identify column types
        numeric_features = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(numeric_features)} numeric features")
        logger.info(f"Identified {len(categorical_features)} categorical features")
        
        preprocessing_metadata['feature_types'] = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
        
        # Combine all preprocessing steps
        preprocessor_steps = []
        
        if numeric_features:
            preprocessor_steps.append(('numeric', numeric_transformer, numeric_features))
        
        if categorical_features:
            preprocessor_steps.append(('categorical', categorical_transformer, categorical_features))
        
        # Create the complete preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop'  # Drop any other columns
        )
        
        # Fit preprocessor on data
        logger.info("Fitting preprocessor on data...")
        self.X = self.df
        self.X_transformed = self.preprocessor.fit_transform(self.df)
        
        # Apply scaling
        scaling_method = self.model_config.get('preprocessing', {}).get('scaling', 'standard')
        preprocessing_metadata['scaling_method'] = scaling_method
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
            logger.info("Applying StandardScaler")
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            logger.info("Applying MinMaxScaler")
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            logger.info("Applying RobustScaler")
        else:
            scaler = None
            logger.info("No scaling applied")
        
        if scaler:
            self.X_scaled = scaler.fit_transform(self.X_transformed)
        else:
            self.X_scaled = self.X_transformed
        
        # Apply dimensionality reduction if selected
        if self.dim_reduction != 'none' and self.X_scaled.shape[1] > self.n_components:
            self.apply_dimensionality_reduction()
            # Use the reduced data for clustering
            X_for_clustering = self.X_reduced
            preprocessing_metadata['dimensionality_reduction'] = {
                'method': self.dim_reduction,
                'components': self.n_components,
                'output_shape': self.X_reduced.shape
            }
        else:
            # Use the scaled data for clustering
            X_for_clustering = self.X_scaled
            preprocessing_metadata['dimensionality_reduction'] = {
                'method': 'none'
            }
        
        preprocessing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        
        # Update metadata
        preprocessing_metadata['preprocessing_time_seconds'] = preprocessing_time
        preprocessing_metadata['output_shape'] = X_for_clustering.shape
        self.metadata['preprocessing'] = preprocessing_metadata
        
        return X_for_clustering
    
    def apply_dimensionality_reduction(self) -> np.ndarray:
        """
        Apply dimensionality reduction to the scaled data.
        
        Returns:
            np.ndarray: Reduced dimensionality data
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        logger.info(f"Applying {self.dim_reduction} dimensionality reduction to {self.n_components} components")
        
        dim_reduction_metadata = {}
        start_time = datetime.now()
        
        # Determine effective n_components
        max_components = min(self.X_scaled.shape[0], self.X_scaled.shape[1])
        n_components = min(self.n_components, max_components)
        
        if n_components < self.n_components:
            logger.warning(f"Requested {self.n_components} components, but data dimensionality allows "
                           f"only {n_components} components.")
            dim_reduction_metadata['requested_components'] = self.n_components
            dim_reduction_metadata['actual_components'] = n_components
        
        # Apply the selected dimensionality reduction method
        if self.dim_reduction == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            self.X_reduced = reducer.fit_transform(self.X_scaled)
            
            # Calculate feature importances (loadings) for PCA
            if hasattr(reducer, 'components_') and hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    loadings = reducer.components_.T * np.sqrt(reducer.explained_variance_)
                    feature_names = self.preprocessor.get_feature_names_out()
                    
                    # Create a DataFrame of feature importances
                    self.feature_importances = pd.DataFrame(
                        loadings, 
                        index=feature_names, 
                        columns=[f'PC{i+1}' for i in range(n_components)]
                    )
                    
                    dim_reduction_metadata['explained_variance'] = {
                        f'PC{i+1}': float(var) for i, var in enumerate(reducer.explained_variance_ratio_)
                    }
                    
                    cumulative_var = np.cumsum(reducer.explained_variance_ratio_)
                    dim_reduction_metadata['cumulative_explained_variance'] = float(cumulative_var[-1])
                    
                    logger.info(f"Cumulative explained variance: {cumulative_var[-1]:.4f}")
                except Exception as e:
                    logger.warning(f"Could not calculate feature importances: {str(e)}")
                
        elif self.dim_reduction == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=self.random_state)
            self.X_reduced = reducer.fit_transform(self.X_scaled)
        
        dim_reduction_time = (datetime.now() - start_time).total_seconds()
        dim_reduction_metadata['time_seconds'] = dim_reduction_time
        dim_reduction_metadata['output_shape'] = self.X_reduced.shape
        
        logger.info(f"Dimensionality reduction completed in {dim_reduction_time:.2f} seconds")
        
        # Update metadata
        self.metadata['preprocessing']['dimensionality_reduction'] = dim_reduction_metadata
        
        return self.X_reduced
    
    def estimate_optimal_clusters(self, data: np.ndarray) -> int:
        """
        Estimate the optimal number of clusters using multiple methods.
        
        Args:
            data: Data to use for estimation
            
        Returns:
            int: Estimated optimal number of clusters
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        from collections import Counter
        
        logger.info("Estimating optimal number of clusters...")
        
        if self.n_clusters is not None:
            logger.info(f"Using user-specified number of clusters: {self.n_clusters}")
            self.optimal_clusters = self.n_clusters
            self.metadata['models']['cluster_estimation'] = {
                'method': 'user_specified',
                'optimal_clusters': self.n_clusters
            }
            return self.n_clusters
        
        estimation_metadata = {}
        start_time = datetime.now()
        
        logger.info(f"Estimating optimal number of clusters in range {self.cluster_range}...")
        estimation_metadata['cluster_range'] = self.cluster_range
        
        # Use the elbow method with K-means to estimate the optimal number of clusters
        inertia_values = []
        silhouette_values = []
        ch_values = []
        db_values = []
        
        max_k = min(self.cluster_range[1], data.shape[0] - 1)
        cluster_range = range(max(self.cluster_range[0], 2), max_k + 1)
        
        logger.info(f"Testing cluster sizes from {cluster_range.start} to {cluster_range.stop - 1}")
        
        # Create dictionaries to store metric values
        metrics_data = {
            'inertia': {},
            'silhouette': {},
            'calinski_harabasz': {},
            'davies_bouldin': {}
        }
        
        for k in cluster_range:
            logger.info(f"Testing k={k} clusters")
            # Run K-means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            inertia_values.append(inertia)
            metrics_data['inertia'][k] = float(inertia)
            
            # Silhouette score
            if 1 < k < data.shape[0]:
                try:
                    silhouette = silhouette_score(data, cluster_labels)
                    silhouette_values.append(silhouette)
                    metrics_data['silhouette'][k] = float(silhouette)
                    logger.debug(f"  k={k}, silhouette={silhouette:.4f}")
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score for k={k}: {str(e)}")
                    silhouette_values.append(0)
                    metrics_data['silhouette'][k] = 0
            else:
                silhouette_values.append(0)
                metrics_data['silhouette'][k] = 0
            
            # Calinski-Harabasz score
            if k > 1:
                try:
                    ch = calinski_harabasz_score(data, cluster_labels)
                    ch_values.append(ch)
                    metrics_data['calinski_harabasz'][k] = float(ch)
                    logger.debug(f"  k={k}, calinski_harabasz={ch:.4f}")
                except Exception as e:
                    logger.warning(f"Could not calculate Calinski-Harabasz score for k={k}: {str(e)}")
                    ch_values.append(0)
                    metrics_data['calinski_harabasz'][k] = 0
            else:
                ch_values.append(0)
                metrics_data['calinski_harabasz'][k] = 0
            
            # Davies-Bouldin score
            if k > 1:
                try:
                    db = davies_bouldin_score(data, cluster_labels)
                    db_values.append(db)
                    metrics_data['davies_bouldin'][k] = float(db)
                    logger.debug(f"  k={k}, davies_bouldin={db:.4f}")
                except Exception as e:
                    logger.warning(f"Could not calculate Davies-Bouldin score for k={k}: {str(e)}")
                    db_values.append(float('inf'))
                    metrics_data['davies_bouldin'][k] = float('inf')
            else:
                db_values.append(float('inf'))
                metrics_data['davies_bouldin'][k] = float('inf')
        
        # Store metric data in metadata
        estimation_metadata['metrics'] = metrics_data
        
        # Find best k based on silhouette score (max)
        if silhouette_values and not all(s == 0 for s in silhouette_values):
            silhouette_best_k = cluster_range[np.argmax(silhouette_values)]
            estimation_metadata['silhouette_method'] = {
                'best_k': int(silhouette_best_k),
                'best_score': float(max(silhouette_values))
            }
        else:
            silhouette_best_k = None
        
        # Find best k based on Calinski-Harabasz score (max)
        if ch_values and not all(ch == 0 for ch in ch_values):
            ch_best_k = cluster_range[np.argmax(ch_values)]
            estimation_metadata['calinski_harabasz_method'] = {
                'best_k': int(ch_best_k),
                'best_score': float(max(ch_values))
            }
        else:
            ch_best_k = None
        
        # Find best k based on Davies-Bouldin score (min)
        if db_values and not all(x == float('inf') for x in db_values):
            db_best_k = cluster_range[np.argmin(db_values)]
            estimation_metadata['davies_bouldin_method'] = {
                'best_k': int(db_best_k),
                'best_score': float(min(db_values))
            }
        else:
            db_best_k = None
        
        # Combine the results from different methods
        best_k_values = [k for k in [silhouette_best_k, ch_best_k, db_best_k] if k is not None]
        
        if best_k_values:
            # Get the most frequent value, or the median if there's a tie
            if len(best_k_values) > 1:
                # Find the most common value
                counter = Counter(best_k_values)
                most_common = counter.most_common()
                
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    # If there's a tie, use the median
                    self.optimal_clusters = int(np.median(best_k_values))
                    estimation_metadata['final_decision'] = {
                        'method': 'median of tied values',
                        'tied_values': list(map(int, best_k_values)),
                        'optimal_clusters': int(self.optimal_clusters)
                    }
                else:
                    # Use the most common value
                    self.optimal_clusters = most_common[0][0]
                    estimation_metadata['final_decision'] = {
                        'method': 'most frequent value',
                        'frequency': int(most_common[0][1]),
                        'all_values': list(map(int, best_k_values)),
                        'optimal_clusters': int(self.optimal_clusters)
                    }
            else:
                # Only one method worked
                self.optimal_clusters = best_k_values[0]
                estimation_metadata['final_decision'] = {
                    'method': 'single valid result',
                    'optimal_clusters': int(self.optimal_clusters)
                }
        else:
            # If all methods fail, use the middle of the range
            self.optimal_clusters = (self.cluster_range[0] + self.cluster_range[1]) // 2
            logger.warning(f"Could not determine optimal clusters. Using {self.optimal_clusters} as default.")
            estimation_metadata['final_decision'] = {
                'method': 'default to middle of range',
                'reason': 'no valid results from any method',
                'optimal_clusters': int(self.optimal_clusters)
            }
        
        total_time = (datetime.now() - start_time).total_seconds()
        estimation_metadata['time_seconds'] = total_time
        
        # Update metadata
        self.metadata['models']['cluster_estimation'] = estimation_metadata
        
        logger.info(f"Optimal number of clusters: {self.optimal_clusters}")
        return self.optimal_clusters
    
    def train_models(self, data=None) -> Dict[str, Any]:
        """
        Train multiple clustering models.
        
        Args:
            data: Preprocessed data for clustering (optional)
            
        Returns:
            dict: Trained models with their labels
        """
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
        from sklearn.mixture import GaussianMixture
        
        logger.info("Training clustering models...")
        
        training_metadata = {}
        start_time = datetime.now()
        
        # Use the provided data or the preprocessed data
        if data is None:
            if self.X_reduced is not None:
                data = self.X_reduced
                logger.info("Using dimensionally-reduced data for clustering")
            else:
                data = self.X_scaled
                logger.info("Using scaled data for clustering")
        
        # Estimate optimal number of clusters if not provided
        if self.optimal_clusters is None:
            self.estimate_optimal_clusters(data)
        
        n_clusters = self.optimal_clusters
        training_metadata['n_clusters'] = n_clusters
        
        # Initialize models dictionary
        self.models = {}
        training_metadata['models'] = {}
        
        # Train K-means
        try:
            logger.info(f"Training K-means with {n_clusters} clusters")
            model_start = datetime.now()
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=self.random_state,
                n_init=10
            )
            kmeans_labels = kmeans.fit_predict(data)
            model_time = (datetime.now() - model_start).total_seconds()
            
            self.models['kmeans'] = {
                'model': kmeans,
                'labels': kmeans_labels
            }
            
            logger.info(f"K-means clustering complete in {model_time:.2f} seconds")
            
            training_metadata['models']['kmeans'] = {
                'trained_successfully': True,
                'time_seconds': model_time,
                'inertia': float(kmeans.inertia_),
                'parameters': {
                    'n_clusters': n_clusters,
                    'random_state': self.random_state,
                    'n_init': 10
                }
            }
        except Exception as e:
            logger.error(f"Error training K-means: {str(e)}", exc_info=True)
            training_metadata['models']['kmeans'] = {
                'trained_successfully': False,
                'error': str(e)
            }
        
        # Train Agglomerative clustering
        try:
            logger.info(f"Training Agglomerative clustering with {n_clusters} clusters")
            model_start = datetime.now()
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            agglomerative_labels = agglomerative.fit_predict(data)
            model_time = (datetime.now() - model_start).total_seconds()
            
            self.models['agglomerative'] = {
                'model': agglomerative,
                'labels': agglomerative_labels
            }
            
            logger.info(f"Agglomerative clustering complete in {model_time:.2f} seconds")
            
            training_metadata['models']['agglomerative'] = {
                'trained_successfully': True,
                'time_seconds': model_time,
                'parameters': {
                    'n_clusters': n_clusters
                }
            }
        except Exception as e:
            logger.error(f"Error training Agglomerative clustering: {str(e)}", exc_info=True)
            training_metadata['models']['agglomerative'] = {
                'trained_successfully': False,
                'error': str(e)
            }
        
        # Train Gaussian Mixture Model
        try:
            logger.info(f"Training Gaussian Mixture Model with {n_clusters} components")
            model_start = datetime.now()
            gmm = GaussianMixture(
                n_components=n_clusters, 
                random_state=self.random_state,
                n_init=10
            )
            gmm_labels = gmm.fit_predict(data)
            model_time = (datetime.now() - model_start).total_seconds()
            
            self.models['gmm'] = {
                'model': gmm,
                'labels': gmm_labels
            }
            
            logger.info(f"Gaussian Mixture Model complete in {model_time:.2f} seconds")
            
            training_metadata['models']['gmm'] = {
                'trained_successfully': True,
                'time_seconds': model_time,
                'parameters': {
                    'n_components': n_clusters,
                    'random_state': self.random_state,
                    'n_init': 10
                }
            }
        except Exception as e:
            logger.error(f"Error training Gaussian Mixture Model: {str(e)}", exc_info=True)
            training_metadata['models']['gmm'] = {
                'trained_successfully': False,
                'error': str(e)
            }
        
        # Train DBSCAN (doesn't require specifying n_clusters)
        try:
            logger.info("Training DBSCAN clustering")
            model_start = datetime.now()
            
            # Estimate eps using nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(20, data.shape[0]-1))
            nn.fit(data)
            distances, _ = nn.kneighbors(data)
            
            # Sort distances to the kth nearest neighbor and use the median as eps
            eps = np.median(np.sort(distances[:, -1]))
            min_samples = min(5, int(data.shape[0] * 0.05) + 1)  # 5% of data points or at least 5
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(data)
            model_time = (datetime.now() - model_start).total_seconds()
            
            # Get the number of clusters (excluding noise)
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = np.sum(dbscan_labels == -1)
            
            self.models['dbscan'] = {
                'model': dbscan,
                'labels': dbscan_labels
            }
            
            logger.info(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points in {model_time:.2f} seconds")
            
            training_metadata['models']['dbscan'] = {
                'trained_successfully': True,
                'time_seconds': model_time,
                'parameters': {
                    'eps': float(eps),
                    'min_samples': int(min_samples)
                },
                'results': {
                    'n_clusters': int(n_clusters_dbscan),
                    'n_noise': int(n_noise),
                    'noise_percentage': float(n_noise/len(dbscan_labels)*100)
                }
            }
        except Exception as e:
            logger.error(f"Error training DBSCAN: {str(e)}", exc_info=True)
            training_metadata['models']['dbscan'] = {
                'trained_successfully': False,
                'error': str(e)
            }
        
        # Train Birch
        try:
            logger.info(f"Training Birch clustering with {n_clusters} clusters")
            model_start = datetime.now()
            birch = Birch(n_clusters=n_clusters)
            birch_labels = birch.fit_predict(data)
            model_time = (datetime.now() - model_start).total_seconds()
            
            self.models['birch'] = {
                'model': birch,
                'labels': birch_labels
            }
            
            logger.info(f"Birch clustering complete in {model_time:.2f} seconds")
            
            training_metadata['models']['birch'] = {
                'trained_successfully': True,
                'time_seconds': model_time,
                'parameters': {
                    'n_clusters': n_clusters
                }
            }
        except Exception as e:
            logger.error(f"Error training Birch: {str(e)}", exc_info=True)
            training_metadata['models']['birch'] = {
                'trained_successfully': False,
                'error': str(e)
            }
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"All clustering models trained in {total_time:.2f} seconds")
        
        training_metadata['total_time_seconds'] = total_time
        training_metadata['n_models_trained'] = len(self.models)
        
        # Update metadata
        self.metadata['models']['training'] = training_metadata
        
        return self.models
    
    def evaluate_models(self, data=None) -> pd.DataFrame:
        """
        Evaluate trained clustering models.
        
        Args:
            data: Data used for clustering (optional)
            
        Returns:
            pd.DataFrame: Results with performance metrics
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        if not self.models:
            error_msg = "No trained models. Call train_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Evaluating clustering models...")
        
        evaluation_metadata = {}
        start_time = datetime.now()
        
        # Use the provided data or the preprocessed data
        if data is None:
            if self.X_reduced is not None:
                data = self.X_reduced
            else:
                data = self.X_scaled
        
        results = []
        
        for name, model_dict in self.models.items():
            try:
                logger.info(f"Evaluating {name} clustering model")
                model_eval_start = datetime.now()
                
                labels = model_dict['labels']
                
                # Count clusters (excluding noise points for DBSCAN)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = np.sum(labels == -1) if -1 in labels else 0
                
                model_metrics = {
                    'n_clusters': int(n_clusters),
                    'n_noise': int(n_noise)
                }
                
                # Calculate metrics (only if more than one cluster)
                if n_clusters > 1:
                    # Silhouette score (ignore noise points)
                    try:
                        if -1 in labels:
                            mask = labels != -1
                            if sum(mask) > n_clusters:  # Need more points than clusters
                                silhouette = silhouette_score(data[mask], labels[mask])
                            else:
                                silhouette = float('nan')
                                logger.warning(f"Not enough non-noise points to calculate silhouette score for {name}")
                        else:
                            silhouette = silhouette_score(data, labels)
                        
                        model_metrics['silhouette'] = float(silhouette)
                    except Exception as e:
                        logger.warning(f"Error calculating silhouette score for {name}: {str(e)}")
                        silhouette = float('nan')
                        model_metrics['silhouette_error'] = str(e)
                    
                    # Calinski-Harabasz index
                    try:
                        if -1 in labels:
                            mask = labels != -1
                            if sum(mask) > n_clusters:
                                ch_score = calinski_harabasz_score(data[mask], labels[mask])
                            else:
                                ch_score = float('nan')
                                logger.warning(f"Not enough non-noise points to calculate Calinski-Harabasz for {name}")
                        else:
                            ch_score = calinski_harabasz_score(data, labels)
                        
                        model_metrics['calinski_harabasz'] = float(ch_score)
                    except Exception as e:
                        logger.warning(f"Error calculating Calinski-Harabasz score for {name}: {str(e)}")
                        ch_score = float('nan')
                        model_metrics['calinski_harabasz_error'] = str(e)
                    
                    # Davies-Bouldin index
                    try:
                        if -1 in labels:
                            mask = labels != -1
                            if sum(mask) > n_clusters:
                                db_score = davies_bouldin_score(data[mask], labels[mask])
                            else:
                                db_score = float('nan')
                                logger.warning(f"Not enough non-noise points to calculate Davies-Bouldin for {name}")
                        else:
                            db_score = davies_bouldin_score(data, labels)
                        
                        model_metrics['davies_bouldin'] = float(db_score)
                    except Exception as e:
                        logger.warning(f"Error calculating Davies-Bouldin score for {name}: {str(e)}")
                        db_score = float('nan')
                        model_metrics['davies_bouldin_error'] = str(e)
                else:
                    logger.warning(f"Model {name} found only {n_clusters} clusters, cannot calculate metrics")
                    silhouette = float('nan')
                    ch_score = float('nan')
                    db_score = float('nan')
                    model_metrics['reason'] = 'insufficient_clusters'
                
                # Store cluster sizes
                cluster_sizes = {}
                for cluster_id in set(labels):
                    count = np.sum(labels == cluster_id)
                    cluster_sizes[int(cluster_id) if cluster_id != -1 else 'noise'] = int(count)
                
                model_metrics['cluster_sizes'] = cluster_sizes
                
                # Store results for DataFrame
                result = {
                    'model': name,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': silhouette if not np.isnan(silhouette) else None,
                    'calinski_harabasz': ch_score if not np.isnan(ch_score) else None,
                    'davies_bouldin': db_score if not np.isnan(db_score) else None
                }
                
                model_eval_time = (datetime.now() - model_eval_start).total_seconds()
                model_metrics['evaluation_time_seconds'] = model_eval_time
                
                results.append(result)
                evaluation_metadata[name] = model_metrics
                
                # Log metrics
                logger.info(f"  {name} evaluation results:")
                logger.info(f"    Clusters: {n_clusters}")
                logger.info(f"    Noise points: {n_noise}")
                
                if not np.isnan(silhouette):
                    logger.info(f"    Silhouette score: {silhouette:.4f}")
                if not np.isnan(ch_score):
                    logger.info(f"    Calinski-Harabasz score: {ch_score:.4f}")
                if not np.isnan(db_score):
                    logger.info(f"    Davies-Bouldin score: {db_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {str(e)}", exc_info=True)
                evaluation_metadata[name] = {
                    'evaluation_error': str(e)
                }
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        
        # Identify best model based on silhouette score (higher is better)
        if not self.results.empty and 'silhouette' in self.results.columns:
            # Make sure to handle None values
            self.results['silhouette'] = pd.to_numeric(self.results['silhouette'], errors='coerce')
            non_nan_results = self.results[~self.results['silhouette'].isna()]
            
            if not non_nan_results.empty:
                best_idx = non_nan_results['silhouette'].idxmax()
                best_model_name = non_nan_results.loc[best_idx, 'model']
                best_silhouette = non_nan_results.loc[best_idx, 'silhouette']
                
                self.best_model = self.models[best_model_name]
                logger.info(f"Best model: {best_model_name} (Silhouette = {best_silhouette:.4f})")
                
                evaluation_metadata['best_model'] = {
                    'name': best_model_name,
                    'silhouette': float(best_silhouette),
                    'criterion': 'silhouette_score'
                }
            else:
                # If all silhouette scores are NaN, choose based on number of clusters
                if 'n_clusters' in self.results.columns:
                    # For no good metric, prefer models with reasonable number of clusters
                    n_clusters_target = self.optimal_clusters if self.optimal_clusters else 5
                    
                    # Find model with number of clusters closest to target
                    self.results['distance_to_target'] = abs(self.results['n_clusters'] - n_clusters_target)
                    best_idx = self.results['distance_to_target'].idxmin()
                    best_model_name = self.results.loc[best_idx, 'model']
                    
                    self.best_model = self.models[best_model_name]
                    logger.info(f"Best model: {best_model_name} (based on having number of clusters closest to target)")
                    
                    evaluation_metadata['best_model'] = {
                        'name': best_model_name,
                        'n_clusters': int(self.results.loc[best_idx, 'n_clusters']),
                        'criterion': 'closest_to_target_clusters'
                    }
                else:
                    # Default to the first model if no valid criteria
                    best_model_name = list(self.models.keys())[0]
                    self.best_model = self.models[best_model_name]
                    logger.warning(f"No valid evaluation criteria. Using {best_model_name} as default best model")
                    
                    evaluation_metadata['best_model'] = {
                        'name': best_model_name,
                        'criterion': 'default_first_model'
                    }
        
        total_time = (datetime.now() - start_time).total_seconds()
        evaluation_metadata['total_time_seconds'] = total_time
        
        # Update metadata
        self.metadata['evaluation'] = evaluation_metadata
        
        return self.results
    
    def save_model(self) -> str:
        """
        Save the best clustering model and preprocessing pipeline.
        
        Returns:
            str: Path to saved model file
        """
        if self.best_model is None:
            error_msg = "No best model selected. Call evaluate_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Saving best model and preprocessing pipeline...")
        
        model_metadata = {}
        start_time = datetime.now()
        
        # Get the best model name
        best_model_name = [k for k, v in self.models.items() if v == self.best_model][0]
        
        # Create a dictionary with all necessary components
        model_package = {
            'preprocessor': self.preprocessor,
            'model': self.best_model['model'],
            'labels': self.best_model['labels'],
            'dim_reduction': self.dim_reduction,
            'n_components': self.n_components,
            'model_name': best_model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        model_filename = os.path.join(self.output_dir, f"{self.model_id}.pkl")
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model package saved to {model_filename}")
        model_metadata['model_file'] = model_filename
        
        # Save model results
        if not self.results.empty:
            results_filename = os.path.join(self.output_dir, f"{self.model_id}_results.csv")
            self.results.to_csv(results_filename, index=False)
            logger.info(f"Results saved to {results_filename}")
            model_metadata['results_file'] = results_filename
        
        total_time = (datetime.now() - start_time).total_seconds()
        model_metadata['save_time_seconds'] = total_time
        
        # Update metadata
        self.metadata['best_model'].update(model_metadata)
        self.save_metadata()
        
        return model_filename

    def run_pipeline(self) -> Tuple[Any, pd.DataFrame]:
        """
        Run the complete pipeline.
        
        Returns:
            tuple: (best_model, evaluation_results)
        """
        logger.info("Starting clustering ML pipeline...")
        pipeline_start = datetime.now()
        
        try:
            # Load and validate data
            logger.info("\nLoading and validating data...")
            self.load_data()
            self.validate_data()
            logger.info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Preprocess data
            logger.info("\nPreprocessing data...")
            data = self.preprocess_data()
            
            # Train models
            logger.info("\nTraining clustering models...")
            self.train_models(data)
            logger.info(f"Trained {len(self.models)} models")
            
            # Evaluate models
            logger.info("\nEvaluating models...")
            self.evaluate_models(data)
            
            # Save the best model
            logger.info("\nSaving model...")
            self.save_model()
            
            # Calculate total runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"\nPipeline completed in {pipeline_runtime:.2f} seconds!")
            
            # Final metadata updates
            self.metadata['runtime_seconds'] = pipeline_runtime
            self.metadata['status'] = 'completed'
            self.save_metadata()
            
            return self.best_model, self.results
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            
            # Update metadata with error information
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.save_metadata()
            
            raise

#########################################
# Time Seies Pipeline
#########################################

class TimeSeriesPipeline(BasePipeline):
   
    def __init__(
            self,
            data_path: Optional[str] = None,
            df: Optional[pd.DataFrame] = None,
            target: Optional[str] = None,
            time_col: Optional[str] = None,
            forecast_horizon: int = 7,
            freq: Optional[str] = None,
            test_size: float = 0.2,
            lag_orders: Optional[List[int]] = None,
            rolling_windows: Optional[List[int]] = None,
            seasonal_period: Optional[int] = None,
            scale_method: str = 'minmax',
            exog_vars: Optional[List[str]] = None,
            config_path: Optional[str] = "config/config.yaml",
            output_dir: Optional[str] = None,
            model_id: Optional[str] = None
        ):
        """
        Initialize the time series pipeline.
        """
        # Call parent class constructor
        super().__init__(data_path=data_path, df=df, target=target, 
                        config_path=config_path, output_dir=output_dir, model_id=model_id)
        
        self.problem_type = 'time_series'
        self.time_col = time_col
        self.forecast_horizon = forecast_horizon
        self.freq = freq
        self.test_size = test_size
        self.lag_orders = lag_orders or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.seasonal_period = seasonal_period
        self.scale_method = scale_method
        self.exog_vars = exog_vars or []
        
        # Get time series-specific configuration
        self.model_config = ml_utils.get_model_config(self.config, 'time_series')
        
        # Initialize time series specific attributes
        self.time_series = None
        self.time_series_train = None
        self.time_series_test = None
        self.train_df = None
        self.test_df = None
        self.raw_df = None
        self.forecast_dates = None
        self.scaler = None
        self.is_stationary = None
        self.stationarity_test_results = None
        self.feature_importance = None
        self.forecast_df = None
        
        logger.info("Time Series Pipeline initialized")
    
    def validate_data(self) -> bool:
        """
        Validate the data for time series analysis.
        """
        logger.info("Validating data for time series analysis...")
        
        validation_metadata = {}
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Store original data
        self.raw_df = self.df.copy()
        
        # Check if target column exists
        if self.target is None:
            error_msg = "Target column name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.target not in self.df.columns:
            error_msg = f"Target column '{self.target}' not found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_metadata['target_exists'] = True
        
        # Check if time column exists
        if self.time_col is None:
            error_msg = "Time column name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.time_col not in self.df.columns:
            error_msg = f"Time column '{self.time_col}' not found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_metadata['time_col_exists'] = True
        
        # Check if time column can be converted to datetime
        try:
            logger.info("Converting time column to datetime")
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
            validation_metadata['time_col_converted'] = True
        except Exception as e:
            error_msg = f"Could not convert time column '{self.time_col}' to datetime: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if target has valid numeric values
        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            error_msg = f"Target column '{self.target}' must contain numeric values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_metadata['target_is_numeric'] = True
        
        # Sort by time if not already sorted
        if not self.df[self.time_col].is_monotonic_increasing:
            logger.warning("Data is not sorted by time. Sorting now.")
            self.df = self.df.sort_values(by=self.time_col)
            validation_metadata['was_sorted'] = True
        else:
            validation_metadata['was_sorted'] = False
        
        # Check for duplicate time values
        dup_count = self.df[self.time_col].duplicated().sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate time values. This may cause issues.")
            validation_metadata['duplicate_times'] = int(dup_count)
        else:
            validation_metadata['duplicate_times'] = 0
        
        # Check for missing values in target
        missing_in_target = self.df[self.target].isnull().sum()
        if missing_in_target > 0:
            logger.warning(f"{missing_in_target} missing values found in target column.")
            validation_metadata['missing_in_target'] = int(missing_in_target)
        else:
            validation_metadata['missing_in_target'] = 0
        
        # Infer frequency if not provided
        if self.freq is None:
            df_temp = self.df.set_index(self.time_col)
            inferred_freq = pd.infer_freq(df_temp.index)
            if inferred_freq:
                self.freq = inferred_freq
                logger.info(f"Inferred frequency: {self.freq}")
                validation_metadata['freq_inferred'] = self.freq
            else:
                logger.warning("Could not infer frequency. Using 'D' (daily) as default.")
                self.freq = 'D'
                validation_metadata['freq_default'] = 'D'
        
        # Set seasonal period if not provided
        if self.seasonal_period is None:
            if self.freq in ['D', 'B']:
                self.seasonal_period = 7  # Weekly seasonality
            elif self.freq in ['W']:
                self.seasonal_period = 52  # Yearly seasonality for weekly data
            elif self.freq in ['M', 'MS']:
                self.seasonal_period = 12  # Yearly seasonality for monthly data
            elif self.freq in ['Q', 'QS']:
                self.seasonal_period = 4  # Yearly seasonality for quarterly data
            elif self.freq in ['H']:
                self.seasonal_period = 24  # Daily seasonality for hourly data
            else:
                self.seasonal_period = 1  # No seasonality
            
            logger.info(f"Set seasonal period to {self.seasonal_period} based on frequency '{self.freq}'")
            validation_metadata['seasonal_period_inferred'] = self.seasonal_period
        
        # Update metadata
        self.metadata['data']['validation'] = validation_metadata
        
        logger.info("Data validation complete")
        return True
    
    def preprocess_data(self):
        """
        Preprocess the data for time series analysis.
        """
        logger.info("Preprocessing data for time series analysis...")
        
        preprocessing_metadata = {}
        
        # Set time column as index
        logger.info(f"Setting {self.time_col} as index")
        df = self.df.copy()
        df.set_index(self.time_col, inplace=True)
        df.sort_index(inplace=True)
        
        # Handle missing values in target
        missing_before = df[self.target].isnull().sum()
        if missing_before > 0:
            logger.info(f"Handling {missing_before} missing values in target")
            # Fill with forward fill, then backward fill
            df[self.target] = df[self.target].fillna(method='ffill').fillna(method='bfill')
            # If still any nulls, fill with mean
            df[self.target] = df[self.target].fillna(df[self.target].mean())
            
            preprocessing_metadata['target_missing_before'] = int(missing_before)
            preprocessing_metadata['target_missing_after'] = int(df[self.target].isnull().sum())
        
        # Create time series and split into train/test
        self.time_series = df[self.target].copy()
        
        train_size = int(len(df) * (1 - self.test_size))
        self.train_df = df.iloc[:train_size].copy()
        self.test_df = df.iloc[train_size:].copy()
        self.time_series_train = self.time_series.iloc[:train_size].copy()
        self.time_series_test = self.time_series.iloc[train_size:].copy()
        
        logger.info(f"Split data: {len(self.time_series_train)} training samples, {len(self.time_series_test)} test samples")
        preprocessing_metadata['train_size'] = len(self.time_series_train)
        preprocessing_metadata['test_size'] = len(self.time_series_test)
        
        # Add date-based features
        feature_extractor = TimeSeriesFeatureExtractor(
            lag_orders=self.lag_orders,
            rolling_windows=self.rolling_windows,
            logger=logger
        )
        
        # Transform both train and test
        df_features = feature_extractor.transform(df)
        
        # Drop rows with NaN (due to lag/rolling features)
        rows_before = df_features.shape[0]
        df_features = df_features.dropna()
        rows_after = df_features.shape[0]
        rows_dropped = rows_before - rows_after
        logger.info(f"Dropped {rows_dropped} rows with NaN values from feature extraction")
        
        # Split features into train/test
        train_features = df_features.iloc[:train_size].copy()
        test_features = df_features.iloc[train_size:].copy()
        
        # Scale features if needed
        if self.scale_method != 'none':
            logger.info(f"Scaling features using {self.scale_method} method")
            
            if self.scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scale_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scale_method == 'robust':
                self.scaler = RobustScaler()
            
            # Scale only numeric columns
            numeric_cols = train_features.select_dtypes(include=np.number).columns
            train_features[numeric_cols] = self.scaler.fit_transform(train_features[numeric_cols])
            test_features[numeric_cols] = self.scaler.transform(test_features[numeric_cols])
            
            preprocessing_metadata['scaling'] = {
                'method': self.scale_method,
                'columns_scaled': len(numeric_cols)
            }
        
        # Prepare for ML models (X=features, y=target)
        self.X_train = train_features.drop(columns=[self.target])
        self.y_train = train_features[self.target]
        self.X_test = test_features.drop(columns=[self.target])
        self.y_test = test_features[self.target]
        
        logger.info(f"Final features shape: X_train={self.X_train.shape}, X_test={self.X_test.shape}")
        preprocessing_metadata['X_train_shape'] = list(self.X_train.shape)
        preprocessing_metadata['X_test_shape'] = list(self.X_test.shape)
        
        # Prepare future dates for forecasting
        last_date = self.test_df.index[-1]
        self.forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=self.freq.lower()),
            periods=self.forecast_horizon,
            freq=self.freq
        )
        
        logger.info(f"Prepared {len(self.forecast_dates)} forecast dates")
        preprocessing_metadata['forecast_dates'] = {
            'start': self.forecast_dates[0].strftime('%Y-%m-%d'),
            'end': self.forecast_dates[-1].strftime('%Y-%m-%d'),
            'count': len(self.forecast_dates)
        }
        
        # Update metadata
        self.metadata['preprocessing'] = preprocessing_metadata
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple time series forecasting models.
        """
        logger.info("Training time series forecasting models...")
        
        training_metadata = {'models': {}}
        
        # Initialize models dictionary
        self.models = {}
        
        # Define ML models to train
        ml_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state)
        }
        
        # Train ML models
        for name, model in ml_models.items():
            try:
                logger.info(f"Training {name} model...")
                model_start = datetime.now()
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(self.y_train, train_pred)
                train_rmse = np.sqrt(train_mse)
                train_mae = mean_absolute_error(self.y_train, train_pred)

                test_mse = mean_squared_error(self.y_test, test_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(self.y_test, test_pred)
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.Series(
                        model.feature_importances_,
                        index=self.X_train.columns
                    ).sort_values(ascending=False)
                elif hasattr(model, 'coef_'):
                    feature_importance = pd.Series(
                        np.abs(model.coef_),
                        index=self.X_train.columns
                    ).sort_values(ascending=False)
                
                # Store the model
                self.models[name] = {
                    'model': model,
                    'predictions': {
                        'train': pd.Series(train_pred, index=self.y_train.index),
                        'test': pd.Series(test_pred, index=self.y_test.index)
                    },
                    'metrics': {
                        'train_mse': train_mse,
                        'train_rmse': train_rmse,
                        'train_mae': train_mae,
                        'test_mse': test_mse,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                    },
                    'feature_importance': feature_importance,
                    'type': 'ml'
                }
                
                model_time = (datetime.now() - model_start).total_seconds()
                
                # Update metadata
                training_metadata['models'][name] = {
                    'type': 'ml',
                    'metrics': {
                        'train_rmse': float(train_rmse),
                        'test_rmse': float(test_rmse),
                    },
                    'training_time_seconds': model_time
                }
                
                logger.info(f"{name} training completed in {model_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error training {name} model: {str(e)}", exc_info=True)
                training_metadata['models'][name] = {
                    'trained_successfully': False,
                    'error': str(e)
                }
        
        # Try ARIMA for time series tasks
        try:
            logger.info("Training ARIMA model...")
            model_start = datetime.now()
            
            # Simple model with defaults
            model = ARIMA(self.time_series_train, order=(1, 1, 1))
            arima_results = model.fit()
            
            # Make predictions
            test_pred = arima_results.forecast(steps=len(self.time_series_test))
            
            test_mse = mean_squared_error(self.time_series_test, test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(self.time_series_test, test_pred)
            
            # Store the model
            self.models['ARIMA'] = {
                'model': arima_results,
                'predictions': {
                    'test': test_pred
                },
                'metrics': {
                    'test_mse': test_mse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                },
                'type': 'statistical'
            }
            
            model_time = (datetime.now() - model_start).total_seconds()
            
            # Update metadata
            training_metadata['models']['ARIMA'] = {
                'type': 'statistical',
                'parameters': {
                    'order': '(1, 1, 1)'
                },
                'metrics': {
                    'test_rmse': float(test_rmse),
                },
                'training_time_seconds': model_time
            }
            
            logger.info(f"ARIMA training completed in {model_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}", exc_info=True)
            training_metadata['models']['ARIMA'] = {
                'trained_successfully': False,
                'error': str(e)
            }
        
        # Update metadata
        training_metadata['n_models_trained'] = len(self.models)
        self.metadata['models']['training'] = training_metadata
        
        logger.info(f"Trained {len(self.models)} time series models")
        
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate trained models and select the best one.
        """
        if not self.models:
            error_msg = "No trained models. Call train_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Evaluating time series models...")
        
        evaluation_metadata = {}
        
        # Collect results
        results = []
        
        for name, model_dict in self.models.items():
            metrics = model_dict['metrics']
            
            results.append({
                'model': name,
                'test_rmse': metrics['test_rmse'],
                'test_mae': metrics['test_mae']
            })
            
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        
        # Sort by test RMSE
        self.results = self.results.sort_values('test_rmse')
        
        # Select best model (lowest test RMSE)
        best_model_name = self.results.iloc[0]['model']
        self.best_model = self.models[best_model_name]
        
        evaluation_metadata['results'] = self.results.to_dict(orient='records')
        evaluation_metadata['best_model'] = best_model_name
        evaluation_metadata['best_metrics'] = {
            'test_rmse': float(self.results.iloc[0]['test_rmse']),
        }
        
        # Update metadata
        self.metadata['evaluation'] = evaluation_metadata
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best model test RMSE: {self.results.iloc[0]['test_rmse']:.4f}")
        
        return self.results
    
    def save_model(self):
        """
        Save the best model and pipeline components.
        """
        if self.best_model is None:
            error_msg = "No best model selected. Call evaluate_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Saving best model and pipeline components...")
        
        model_metadata = {}
        
        # Get the best model name
        best_model_name = self.results.iloc[0]['model']
        
        # Save components needed for prediction
        pipeline_components = {
            'best_model': self.best_model,
            'best_model_name': best_model_name,
            'scaler': self.scaler,
            'target': self.target,
            'time_col': self.time_col,
            'freq': self.freq,
            'seasonal_period': self.seasonal_period,
            'last_date': self.time_series.index[-1],
            'last_values': self.time_series.tail(max(self.lag_orders)).values,
            'lag_orders': self.lag_orders,
            'rolling_windows': self.rolling_windows,
            'X_train_columns': self.X_train.columns.tolist()
        }
        
        # Save model package
        model_filename = os.path.join(self.output_dir, f"{self.model_id}.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(pipeline_components, f)
        
        model_metadata['model_file'] = model_filename
        model_metadata['best_model_name'] = best_model_name
        model_metadata['best_model_type'] = self.best_model['type']
        
        # Update metadata
        self.metadata['best_model'].update(model_metadata)
        
        logger.info(f"Time series model saved to {model_filename}")
        
        return model_filename

    def run_pipeline(self):
        """
        Run the complete pipeline.
        """
        logger.info("Starting time series pipeline...")
        pipeline_start = datetime.now()
        
        try:
            # Load and validate data
            logger.info("\nLoading and validating data...")
            self.load_data()
            self.validate_data()
            
            # Preprocess data
            logger.info("\nPreprocessing data...")
            self.preprocess_data()
            
            # Train models
            logger.info("\nTraining models...")
            self.train_models()
            
            # Evaluate models
            logger.info("\nEvaluating models...")
            self.evaluate_models()
            
            # Save the best model
            logger.info("\nSaving model...")
            self.save_model()
            
            # Calculate total runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"\nPipeline completed in {pipeline_runtime:.2f} seconds!")
            
            # Final metadata updates
            self.metadata['runtime_seconds'] = pipeline_runtime
            self.metadata['status'] = 'completed'
            self.save_metadata()
            
            return self.best_model, self.results
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            
            # Update metadata with error information
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.save_metadata()
            
            raise