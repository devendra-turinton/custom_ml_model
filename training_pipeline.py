import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, Callable, List
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import src.ml_utils as ml_utils

# Get the logger
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

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
        # Log initialization start
        logger.info("Initializing pipeline...")
        
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
        logger.debug(f"Created output directory: {self.output_dir}")
        
        # Load configuration
        try:
            logger.info(f"Loading configuration from {config_path}")
            self.config = ml_utils.load_config(config_path)
            logger.debug(f"Configuration loaded successfully: {json.dumps(self.config, indent=2, default=str)}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {str(e)}. Using defaults.", exc_info=True)
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
        logger.debug(f"Train-test split parameters: test_size={self.test_size}, random_state={self.random_state}")
        
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
        
        # Log system resources at initialization
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            logger.info(f"System resources at initialization: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
        except ImportError:
            logger.debug("psutil not available for resource monitoring")
        
        logger.info(f"{self.__class__.__name__} initialized - ID: {self.model_id}, Version: {self.version}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file or use provided DataFrame.
        
        Returns:
            DataFrame: Loaded data
        """
        logger.info("Loading data...")
        start_time = datetime.now()
        
        if self.df is not None:
            logger.info("Using provided DataFrame")
            self.metadata['data']['source'] = 'provided_dataframe'
            self.metadata['data']['shape'] = self.df.shape
            
            # Log data summary information
            logger.debug(f"Data summary:\n{self._get_data_summary(self.df)}")
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Using provided DataFrame (rows: {self.df.shape[0]:,}, columns: {self.df.shape[1]:,}) in {load_time:.2f} seconds")
            
            return self.df
        
        if self.data_path is None:
            error_msg = "Either data_path or df must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        file_ext = os.path.splitext(self.data_path)[1].lower()
        self.metadata['data']['source'] = self.data_path
        self.metadata['data']['file_extension'] = file_ext
        
        try:
            logger.info(f"Loading data from {self.data_path} (format: {file_ext})")
            
            if file_ext == '.csv':
                logger.info(f"Loading CSV file: {self.data_path}")
                self.df = pd.read_csv(self.data_path)
                logger.debug(f"CSV file loaded successfully with {self.df.shape[0]:,} rows and {self.df.shape[1]:,} columns")
            elif file_ext in ['.xls', '.xlsx']:
                logger.info(f"Loading Excel file: {self.data_path}")
                self.df = pd.read_excel(self.data_path)
                logger.debug(f"Excel file loaded successfully with {self.df.shape[0]:,} rows and {self.df.shape[1]:,} columns")
            elif file_ext == '.json':
                logger.info(f"Loading JSON file: {self.data_path}")
                self.df = pd.read_json(self.data_path)
                logger.debug(f"JSON file loaded successfully with {self.df.shape[0]:,} rows and {self.df.shape[1]:,} columns")
            elif file_ext == '.parquet':
                logger.info(f"Loading Parquet file: {self.data_path}")
                self.df = pd.read_parquet(self.data_path)
                logger.debug(f"Parquet file loaded successfully with {self.df.shape[0]:,} rows and {self.df.shape[1]:,} columns")
            else:
                error_msg = f"Unsupported file extension: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data loaded in {load_time:.2f} seconds")
            
            # Calculate and log memory usage
            memory_usage_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"DataFrame memory usage: {memory_usage_mb:.2f} MB")
            
            # Log data shape and sample
            logger.info(f"Data shape: {self.df.shape[0]:,} rows, {self.df.shape[1]:,} columns")
            logger.debug(f"Data head (first 5 rows):\n{self.df.head().to_string()}")
            
            # Update metadata
            self.metadata['data']['shape'] = self.df.shape
            self.metadata['data']['loading_time_seconds'] = load_time
            self.metadata['data']['memory_usage_mb'] = memory_usage_mb
            self.metadata['data']['columns'] = list(self.df.columns)
            self.metadata['data']['dtypes'] = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            
            # Calculate basic statistics for numeric columns
            numeric_columns = self.df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                # Basic descriptive statistics
                self.metadata['data']['numeric_stats'] = self.df[numeric_columns].describe().to_dict()
                logger.debug(f"Numeric column statistics:\n{self.df[numeric_columns].describe().to_string()}")
            
            # Track missing values
            missing_values = self.df.isnull().sum().to_dict()
            missing_values_percent = {col: self.df[col].isnull().mean() * 100 for col in self.df.columns}
            self.metadata['data']['missing_values'] = missing_values
            self.metadata['data']['missing_values_percent'] = missing_values_percent
            
            # Log columns with missing values
            columns_with_missing = [col for col, count in missing_values.items() if count > 0]
            if columns_with_missing:
                logger.info(f"Found {len(columns_with_missing)} columns with missing values")
                for col in columns_with_missing:
                    logger.info(f"  - Column '{col}': {missing_values[col]:,} missing values ({missing_values_percent[col]:.2f}%)")
            else:
                logger.info("No missing values found in the dataset")
            
            # Log data summary
            logger.debug(f"Data summary:\n{self._get_data_summary(self.df)}")
            
        except Exception as e:
            error_msg = f"Error loading data from {self.data_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
        return self.df
    
    def _get_data_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a summary of the dataframe for logging purposes.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            str: Summary of the dataframe
        """
        summary = []
        summary.append(f"Shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        
        # Data types summary
        dtypes_count = df.dtypes.value_counts().to_dict()
        summary.append("Data types:")
        for dtype, count in dtypes_count.items():
            summary.append(f"  - {dtype}: {count:,} columns")
        
        # Missing values summary
        missing_cols = df.columns[df.isnull().any()].tolist()
        summary.append(f"Columns with missing values: {len(missing_cols):,}")
        
        # Sample values for each column type
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            sample_col = numeric_cols[0]
            summary.append(f"Sample numeric column '{sample_col}' min: {df[sample_col].min()}, max: {df[sample_col].max()}, mean: {df[sample_col].mean():.2f}")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            sample_col = categorical_cols[0]
            unique_values = df[sample_col].nunique()
            summary.append(f"Sample categorical column '{sample_col}' has {unique_values:,} unique values")
        
        return "\n".join(summary)
    
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
        start_time = datetime.now()
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info(f"Examining target column: '{self.target}'")
            # Log target column properties for debugging
            if self.target in self.df.columns:
                target_dtype = self.df[self.target].dtype
                unique_values = self.df[self.target].nunique()
                is_numeric = pd.api.types.is_numeric_dtype(self.df[self.target])
                
                logger.debug(f"Target column properties - dtype: {target_dtype}, unique values: {unique_values}, is_numeric: {is_numeric}")
                logger.debug(f"Target value distribution:\n{self.df[self.target].value_counts().head(10).to_string()}")
                
                if is_numeric:
                    logger.debug(f"Target numeric statistics:\n{self.df[self.target].describe().to_string()}")
            else:
                logger.error(f"Target column '{self.target}' not found in dataframe")
                raise ValueError(f"Target column '{self.target}' not found in dataframe")
            
            self.problem_type = ml_utils.detect_problem_type(self.df, self.target, self.config)
            
            detection_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Detected problem type: {self.problem_type} in {detection_time:.2f} seconds")
            
            # Save problem type and target column to metadata
            self.metadata['parameters']['problem_type'] = self.problem_type
            self.metadata['parameters']['target_column'] = self.target
            
            # Log more details about the detected problem
            if self.problem_type == 'classification':
                # For classification, log class distribution
                class_distribution = self.df[self.target].value_counts(normalize=True) * 100
                logger.info(f"Class distribution in target '{self.target}':")
                for cls, pct in class_distribution.items():
                    logger.info(f"  - {cls}: {pct:.2f}%")
                
                # Check class imbalance
                if class_distribution.min() < 10:  # Arbitrary threshold of 10%
                    logger.warning(f"Class imbalance detected: Minority class represents only {class_distribution.min():.2f}% of the data")
                    
            elif self.problem_type == 'regression':
                # For regression, log distribution statistics
                target_stats = self.df[self.target].describe()
                logger.info(f"Target '{self.target}' regression statistics:")
                logger.info(f"  - Range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")
                logger.info(f"  - Mean: {target_stats['mean']:.4f}, Std: {target_stats['std']:.4f}")
                logger.info(f"  - Quartiles: 25%={target_stats['25%']:.4f}, 50%={target_stats['50%']:.4f}, 75%={target_stats['75%']:.4f}")
            
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
        overall_start_time = datetime.now()
        
        preprocessing_metadata = {}
        
        # Split into features and target
        logger.debug("Splitting data into features and target")
        start_time = datetime.now()
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        split_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Split data into features and target in {split_time:.2f} seconds")
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        preprocessing_metadata['initial_shape'] = {'X': X.shape, 'y': y.shape}
        preprocessing_metadata['target_column'] = self.target
        
        # Apply custom feature engineering if provided
        if custom_feature_engineering:
            logger.info("Applying custom feature engineering...")
            start_time = datetime.now()
            try:
                X_before = X.shape
                X = custom_feature_engineering(X, self.config)
                X_after = X.shape
                engineering_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"Custom feature engineering completed in {engineering_time:.2f} seconds")
                logger.info(f"Features shape before engineering: {X_before}, after: {X_after}")
                
                # Log new features if any were added
                if X_after[1] > X_before[1]:
                    new_features = set(X.columns) - set(self.df.columns)
                    logger.info(f"Added {len(new_features)} new features: {', '.join(new_features)}")
                
                preprocessing_metadata['custom_features_applied'] = True
                preprocessing_metadata['features_after_engineering'] = X.shape
                preprocessing_metadata['feature_engineering_time_seconds'] = engineering_time
            except Exception as e:
                error_msg = f"Error in custom feature engineering: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.warning("Continuing with original features")
                preprocessing_metadata['custom_features_applied'] = False
                preprocessing_metadata['custom_features_error'] = str(e)
        
        # Identify timestamp columns
        logger.debug("Identifying timestamp columns")
        start_time = datetime.now()
        timestamp_columns = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                timestamp_columns.append(col)
                logger.debug(f"Found datetime column: {col}")
            # Try to convert string columns that might be timestamps
            elif X[col].dtype == 'object':
                try:
                    sample = X[col].dropna().iloc[0] if not X[col].dropna().empty else None
                    if sample:
                        logger.debug(f"Attempting to convert column '{col}' to datetime. Sample value: {sample}")
                    pd.to_datetime(X[col], errors='raise')
                    # Convert to actual datetime
                    X[col] = pd.to_datetime(X[col])
                    timestamp_columns.append(col)
                    logger.debug(f"Successfully converted '{col}' to datetime")
                except Exception as e:
                    logger.debug(f"Column '{col}' is not a datetime: {str(e)}")
                    pass
        
        timestamp_detection_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Identified {len(timestamp_columns)} timestamp columns in {timestamp_detection_time:.2f} seconds")
        if timestamp_columns:
            logger.info(f"Timestamp columns: {', '.join(timestamp_columns)}")
        preprocessing_metadata['timestamp_columns'] = timestamp_columns
        
        # Get stratify parameter based on problem type
        stratify = None
        if self.problem_type == 'classification' and self.config.get('common', {}).get('train_test_split', {}).get('stratify', True):
            stratify = y
            logger.info("Using stratified sampling for train-test split")
        
        # Split into train and test sets
        logger.info(f"Splitting data with test_size={self.test_size}, random_state={self.random_state}")
        start_time = datetime.now()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        split_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Train-test split completed in {split_time:.2f} seconds")
        logger.info(f"Train set size: {self.X_train.shape[0]:,} rows, Test set size: {self.X_test.shape[0]:,} rows")
        
        # Log class distribution in train/test if classification
        if self.problem_type == 'classification':
            train_class_dist = pd.Series(self.y_train).value_counts(normalize=True) * 100
            test_class_dist = pd.Series(self.y_test).value_counts(normalize=True) * 100
            
            logger.info("Class distribution in train set:")
            for cls, pct in train_class_dist.items():
                logger.info(f"  - {cls}: {pct:.2f}%")
            
            logger.info("Class distribution in test set:")
            for cls, pct in test_class_dist.items():
                logger.info(f"  - {cls}: {pct:.2f}%")
        
        preprocessing_metadata['split_sizes'] = {
            'train': {'X': self.X_train.shape, 'y': self.y_train.shape},
            'test': {'X': self.X_test.shape, 'y': self.y_test.shape},
        }
        preprocessing_metadata['split_time_seconds'] = split_time
        
        # Create preprocessing steps for different column types
        logger.info("Creating preprocessing pipeline...")
        
        # Get preprocessing config
        preproc_config = self.config.get('common', {}).get('preprocessing', {})
        outlier_config = preproc_config.get('outlier_detection', {})
        missing_values_config = preproc_config.get('missing_values', {})
        feature_eng_config = self.config.get('common', {}).get('feature_engineering', {})
        
        # Log preprocessing configuration
        logger.debug(f"Preprocessing configuration:")
        logger.debug(f"  - Missing values strategies: numeric={missing_values_config.get('numeric_strategy', 'median')}, categorical={missing_values_config.get('categorical_strategy', 'most_frequent')}")
        logger.debug(f"  - Outlier handling: method={outlier_config.get('method', 'iqr')}, threshold={outlier_config.get('threshold', 1.5)}, strategy={outlier_config.get('strategy', 'clip')}")
        logger.debug(f"  - Scaling method: {feature_eng_config.get('scaling', 'standard')}")
        
        # Numeric transformer pipeline
        start_time = datetime.now()
        logger.debug("Creating numeric transformation pipeline")
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
                logger.debug("Added StandardScaler to numeric pipeline")
            elif scaling_method == 'minmax':
                numeric_transformer.steps.append(('scaler', MinMaxScaler()))
                logger.debug("Added MinMaxScaler to numeric pipeline")
        
        numeric_pipeline_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Created numeric transformation pipeline in {numeric_pipeline_time:.2f} seconds")
        
        # Categorical transformer
        start_time = datetime.now()
        logger.debug("Creating categorical transformation pipeline")
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=missing_values_config.get('categorical_strategy', 'most_frequent'))),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        categorical_pipeline_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Created categorical transformation pipeline in {categorical_pipeline_time:.2f} seconds")
        
        # Identify column types
        logger.debug("Identifying column types")
        start_time = datetime.now()
        numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical_features = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        
        column_type_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Identified column types in {column_type_time:.2f} seconds:")
        logger.info(f"  - Numeric features: {len(numeric_features):,}")
        logger.info(f"  - Categorical features: {len(categorical_features):,}")
        
        # Log some sample column names of each type
        if numeric_features:
            logger.debug(f"Sample numeric features (max 5): {', '.join(numeric_features[:5])}")
        if categorical_features:
            logger.debug(f"Sample categorical features (max 5): {', '.join(categorical_features[:5])}")
        
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
        logger.debug("Creating complete preprocessing pipeline")
        start_time = datetime.now()
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop'  # Drop any other columns
        )
        preprocessor_creation_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Created complete preprocessor in {preprocessor_creation_time:.2f} seconds")
        
        # Fit preprocessor on training data
        logger.info("Fitting preprocessor on training data...")
        start_time = datetime.now()
        try:
            self.preprocessor.fit(self.X_train)
            fit_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Preprocessor fitted successfully in {fit_time:.2f} seconds")
            
            # Log transformed feature names if available
            try:
                feature_names_out = self.preprocessor.get_feature_names_out()
                logger.debug(f"Transformed feature count: {len(feature_names_out)}")
                if len(feature_names_out) <= 20:  # Only log all names if not too many
                    logger.debug(f"Transformed feature names: {', '.join(feature_names_out)}")
                else:
                    logger.debug(f"First 10 transformed feature names: {', '.join(feature_names_out[:10])}")
                
                preprocessing_metadata['transformed_feature_count'] = len(feature_names_out)
            except Exception as e:
                logger.debug(f"Could not extract transformed feature names: {str(e)}")
            
            # Track one-hot encoding expansion
            categorical_transformer_idx = None
            for i, (name, _, _) in enumerate(preprocessor_steps):
                if name == 'categorical':
                    categorical_transformer_idx = i
                    break
                    
            if categorical_transformer_idx is not None and categorical_features:
                try:
                    cat_encoder = self.preprocessor.transformers_[categorical_transformer_idx][1].named_steps['encoder']
                    categories = cat_encoder.categories_
                    total_encoded_features = sum(len(c) for c in categories)
                    logger.info(f"One-hot encoding expanded {len(categorical_features)} categorical features into {total_encoded_features} binary features")
                    preprocessing_metadata['one_hot_encoding'] = {
                        'input_features': len(categorical_features),
                        'output_features': total_encoded_features
                    }
                except Exception as e:
                    logger.debug(f"Could not extract one-hot encoding details: {str(e)}")
                
        except Exception as e:
            error_msg = f"Error fitting preprocessor: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
        # Record preprocessing time
        preprocessing_time = (datetime.now() - overall_start_time).total_seconds()
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
        start_time = datetime.now()
        
        # Get model name
        best_model_name = self.metadata['best_model'].get('name', 'unknown')
        
        # Create a proper scikit-learn pipeline from components
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.best_model)
        ])
        
        # Use model_id as the filename
        model_filename = os.path.join(self.output_dir, f"{self.model_id}.pkl")
        
        # Log model size before saving
        try:
            import sys
            model_size_bytes = sys.getsizeof(pickle.dumps(pipeline))
            model_size_mb = model_size_bytes / (1024 * 1024)
            logger.info(f"Model size: {model_size_mb:.2f} MB")
        except Exception as e:
            logger.debug(f"Could not determine model size: {str(e)}")
        
        # Save model
        try:
            with open(model_filename, 'wb') as f:
                pickle.dump(pipeline, f)
            
            save_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Best model ({best_model_name}) saved to {model_filename} in {save_time:.2f} seconds")
            
            # Verify the file was created
            if os.path.exists(model_filename):
                file_size_bytes = os.path.getsize(model_filename)
                file_size_mb = file_size_bytes / (1024 * 1024)
                logger.info(f"Saved model file size: {file_size_mb:.2f} MB")
            else:
                logger.warning(f"Model file not found after saving: {model_filename}")
            
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
        # Update metadata
        self.metadata['best_model']['filename'] = model_filename
        self.metadata['best_model']['file_size_bytes'] = os.path.getsize(model_filename)
        self.metadata['best_model']['save_time_seconds'] = save_time
        
        return model_filename
    
    def save_metadata(self) -> str:
        """
        Save metadata to file, with option to split large prediction data.
        
        Returns:
            str: Path to saved metadata file
        """
        logger.info("Saving metadata...")
        start_time = datetime.now()
        
        metadata_copy = self.metadata.copy()
        predictions_samples = metadata_copy.get('best_model', {}).get('prediction_samples', [])
        
        if len(predictions_samples) > 1000:
            logger.info(f"Large number of prediction samples detected ({len(predictions_samples)}), saving to separate file")
            
            predictions_file = os.path.join(self.output_dir, f"{self.model_id}_prediction_samples.json")
            with open(predictions_file, 'w') as f:
                json.dump(predictions_samples, f, indent=2)
            
            metadata_copy['best_model']['prediction_samples'] = f"Saved separately to {predictions_file}"
            metadata_copy['best_model']['prediction_samples_file'] = predictions_file
            logger.info(f"Prediction samples saved to {predictions_file}")
        
        # Add timestamp to metadata
        metadata_copy['save_timestamp'] = datetime.now().isoformat()
        
        # Save the metadata
        metadata_file = ml_utils.save_metadata(
            metadata_copy,
            self.output_dir,
            filename="metadata.json"
        )
        
        save_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Metadata saved to {metadata_file} in {save_time:.2f} seconds")
        
        return metadata_file

    def run_pipeline(self) -> Tuple[Any, pd.DataFrame]:
        """
        Run the complete pipeline.
        
        Returns:
            tuple: (best_model, evaluation_results)
        """
        logger.info(f"Starting ML pipeline run - ID: {self.model_id}, Version: {self.version}")
        pipeline_start = datetime.now()
        
        # Log system resources at pipeline start
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            logger.info(f"System resources at pipeline start: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
        except ImportError:
            logger.debug("psutil not available for resource monitoring")
        
        try:
            # Load and validate data
            logger.info("\n" + "="*50)
            logger.info("STEP 1: Loading and validating data")
            logger.info("="*50)
            step_start = datetime.now()
            self.load_data()
            
            # Auto-detect problem type if not explicitly set
            if self.problem_type is None:
                self.detect_problem_type()
            
            self.metadata['parameters']['problem_type'] = self.problem_type

            # Validate data based on problem type
            self.validate_data()
            step_time = (datetime.now() - step_start).total_seconds()
            logger.info(f"Data loaded and validated: {self.df.shape[0]:,} rows, {self.df.shape[1]:,} columns")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            # Preprocess data
            logger.info("\n" + "="*50)
            logger.info("STEP 2: Preprocessing data")
            logger.info("="*50)
            step_start = datetime.now()
            self.preprocess_data()
            step_time = (datetime.now() - step_start).total_seconds()
            logger.info(f"Train set: {self.X_train.shape[0]:,} rows, Test set: {self.X_test.shape[0]:,} rows")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            # Train models
            logger.info("\n" + "="*50)
            logger.info("STEP 3: Training models")
            logger.info("="*50)
            step_start = datetime.now()
            self.train_models()
            step_time = (datetime.now() - step_start).total_seconds()
            logger.info(f"Trained {len(self.models):,} models")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            # Evaluate models
            logger.info("\n" + "="*50)
            logger.info("STEP 4: Evaluating models")
            logger.info("="*50)
            step_start = datetime.now()
            self.evaluate_models()
            step_time = (datetime.now() - step_start).total_seconds()
            
            # Log best model details
            best_model_name = self.metadata['best_model'].get('name', 'Unknown')
            best_model_metric = self.metadata['best_model'].get('primary_metric', 'Unknown')
            best_model_score = self.metadata['best_model'].get('primary_metric_value', 'Unknown')
            logger.info(f"Best model: {best_model_name} with {best_model_metric}={best_model_score}")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            # Save the best model
            logger.info("\n" + "="*50)
            logger.info("STEP 5: Saving model")
            logger.info("="*50)
            step_start = datetime.now()
            model_path = self.save_model()
            step_time = (datetime.now() - step_start).total_seconds()
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            # Calculate total runtime
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            
            # Log system resources at pipeline end
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                logger.info(f"System resources at pipeline end: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
            except ImportError:
                logger.debug("psutil not available for resource monitoring")
            
            logger.info("\n" + "="*50)
            logger.info(f"Pipeline completed successfully in {pipeline_runtime:.2f} seconds!")
            logger.info("="*50)
            
            # Final metadata updates
            self.metadata['runtime_seconds'] = pipeline_runtime
            self.metadata['status'] = 'completed'
            self.save_metadata()
            
            return self.best_model, self.results
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            
            # Log exception details
            import traceback
            error_details = traceback.format_exc()
            logger.debug(f"Exception traceback:\n{error_details}")
            
            # Log system resources at failure
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                logger.info(f"System resources at failure: CPU {cpu_percent}%, RAM {memory_info.percent}% (Available: {memory_info.available / (1024**3):.2f} GB)")
            except ImportError:
                logger.debug("psutil not available for resource monitoring")
            
            # Update metadata with error information
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['error_traceback'] = error_details
            self.metadata['failure_timestamp'] = datetime.now().isoformat()
            self.save_metadata()
            
            # Log pipeline failure
            pipeline_runtime = (datetime.now() - pipeline_start).total_seconds()
            logger.error(f"Pipeline failed after {pipeline_runtime:.2f} seconds")
            
            raise