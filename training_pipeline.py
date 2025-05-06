import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, Callable
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import src.ml_utils as ml_utils
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
