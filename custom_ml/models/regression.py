from datetime import datetime
import numpy as np
import pandas as pd
import logging
import os
import traceback
from typing import Any, Dict, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from custom_ml.src import ml_utils
from custom_ml.training_pipeline import BasePipeline
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import logging



#########################################
# Regression Pipeline
#########################################

class RegressionPipeline(BasePipeline):
    
    def __init__(
            self, 
            **kwargs
        ):
        """
        Initialize the regression pipeline.
        
        Args:
            custom_models_function: Optional function that creates custom models
            custom_features_function: Optional function for feature engineering
            **kwargs: Other arguments passed to parent class
        """
        logger.info("Initializing Regression Pipeline...")
        init_start_time = datetime.now()
        
        # Log input arguments except for df
        log_kwargs = {k: v for k, v in kwargs.items() if k != 'df'}
        logger.debug(f"Initialization parameters: {log_kwargs}")
        
        super().__init__(**kwargs)
        self.problem_type = 'regression'
        
        # Get regression-specific configuration
        config_start_time = datetime.now()
        self.model_config = ml_utils.get_model_config(self.config, 'regression')
        config_time = (datetime.now() - config_start_time).total_seconds()
        
        # Log configuration details
        if self.model_config:
            enabled_models = self.model_config.get('models', {}).get('enabled', [])
            logger.debug(f"Loaded regression configuration in {config_time:.2f} seconds")
            logger.debug(f"Enabled models: {enabled_models}")
            
            # Log random search configuration if it exists
            if 'random_search' in self.model_config and self.model_config['random_search'].get('enabled', False):
                rs_config = self.model_config['random_search']
                logger.debug(f"Random search enabled: {rs_config.get('n_iter', 20)} iterations, {rs_config.get('cv', 5)}-fold CV")
                
            # Log evaluation metrics
            eval_config = self.model_config.get('evaluation', {})
            metrics = eval_config.get('metrics', ['r2', 'rmse', 'mae'])
            primary_metric = eval_config.get('primary_metric', 'r2')
            logger.debug(f"Evaluation metrics: {metrics}, primary: {primary_metric}")
        else:
            logger.warning("No regression-specific configuration found, using defaults")
        
        # Log DataFrame summary if provided
        if 'df' in kwargs and kwargs['df'] is not None:
            df = kwargs['df']
            logger.debug(f"DataFrame provided: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            
            # Log data types summary
            dtype_counts = df.dtypes.value_counts().to_dict()
            logger.debug(f"DataFrame column types: {dtype_counts}")
            
            # Log target column info if provided
            if 'target' in kwargs and kwargs['target'] is not None:
                target = kwargs['target']
                if target in df.columns:
                    # Basic target statistics
                    if pd.api.types.is_numeric_dtype(df[target]):
                        target_stats = df[target].describe()
                        logger.debug(f"Target column '{target}' statistics: min={target_stats['min']:.4g}, max={target_stats['max']:.4g}, mean={target_stats['mean']:.4g}")
                    else:
                        logger.warning(f"Target column '{target}' is not numeric: {df[target].dtype}")
                else:
                    logger.warning(f"Target column '{target}' not found in DataFrame columns")
        
        init_time = (datetime.now() - init_start_time).total_seconds()
        logger.info(f"Regression Pipeline initialized in {init_time:.2f} seconds")
        
        # Log memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            logger.debug(f"Memory usage after initialization: {memory_mb:.2f} MB")
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
    
    def validate_data(self) -> bool:
        """
        Validate the data for regression tasks.
        
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating data for regression...")
        validation_start_time = datetime.now()
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.target is None:
            error_msg = "Target column name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Data overview
        logger.debug(f"DataFrame shape: {self.df.shape[0]:,} rows, {self.df.shape[1]:,} columns")
        logger.debug(f"Target column: '{self.target}'")
        
        # Validation checks
        validation_results = {}
        
        # Check if target column exists
        logger.debug(f"Checking if target column '{self.target}' exists in data...")
        if self.target not in self.df.columns:
            error_msg = f"Target column '{self.target}' not found in data"
            logger.error(error_msg)
            logger.debug(f"Available columns: {list(self.df.columns)}")
            raise ValueError(error_msg)
        
        validation_results['target_exists'] = True
        logger.debug(f"Target column '{self.target}' exists in data ✓")
        
        # Check if dataset is empty
        logger.debug("Checking if dataset is empty...")
        if len(self.df) == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['dataset_not_empty'] = True
        logger.debug(f"Dataset is not empty, contains {len(self.df):,} rows ✓")
        
        # Check if target has valid numeric values
        logger.debug(f"Checking if target column '{self.target}' contains numeric values...")
        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            error_msg = f"Target column '{self.target}' must contain numeric values for regression"
            logger.error(error_msg)
            logger.debug(f"Target data type: {self.df[self.target].dtype}")
            # Try to provide example values
            try:
                examples = self.df[self.target].head(3).tolist()
                logger.debug(f"Example values: {examples}")
            except:
                pass
            raise ValueError(error_msg)
        
        validation_results['target_is_numeric'] = True
        logger.debug(f"Target column '{self.target}' contains numeric values ✓")
        
        # Check for at least some valid values in target
        logger.debug("Checking for valid values in target column...")
        if self.df[self.target].isna().all():
            error_msg = f"Target column '{self.target}' contains only NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['target_has_values'] = True
        logger.debug("Target column contains valid values ✓")
        
        # Additional checks
        # Number of missing values in target
        missing_in_target = self.df[self.target].isna().sum()
        missing_pct = (missing_in_target / len(self.df)) * 100
        validation_results['missing_in_target'] = int(missing_in_target)
        validation_results['missing_in_target_pct'] = float(missing_pct)
        
        if missing_in_target > 0:
            logger.warning(f"Missing values in target: {missing_in_target:,} ({missing_pct:.2f}% of rows)")
        else:
            logger.debug("No missing values in target ✓")
        
        # Feature analysis
        logger.debug("Analyzing features for regression...")
        
        # Check for numeric features
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target]
        validation_results['numeric_feature_count'] = len(numeric_cols)
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric features found (excluding target)")
        else:
            logger.debug(f"Found {len(numeric_cols):,} numeric features")
            
            # Check for missing values in numeric features
            numeric_missing = self.df[numeric_cols].isna().sum().sum()
            numeric_missing_pct = (numeric_missing / (len(self.df) * len(numeric_cols))) * 100
            validation_results['numeric_features_missing'] = int(numeric_missing)
            validation_results['numeric_features_missing_pct'] = float(numeric_missing_pct)
            
            if numeric_missing > 0:
                logger.debug(f"Missing values in numeric features: {numeric_missing:,} ({numeric_missing_pct:.2f}% of cells)")
                
                # Report columns with highest missing percentages
                missing_by_col = self.df[numeric_cols].isna().mean() * 100
                high_missing = missing_by_col[missing_by_col > 25].sort_values(ascending=False)
                if not high_missing.empty:
                    logger.warning(f"Features with >25% missing values: {dict(high_missing)}")
        
        # Check for categorical features
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        validation_results['categorical_feature_count'] = len(cat_cols)
        
        if len(cat_cols) > 0:
            logger.debug(f"Found {len(cat_cols):,} categorical features")
            
            # Check cardinality of categorical features
            high_cardinality_cols = []
            for col in cat_cols:
                nunique = self.df[col].nunique()
                if nunique > 100:  # Arbitrary threshold
                    high_cardinality_cols.append((col, nunique))
            
            if high_cardinality_cols:
                logger.warning(f"High cardinality categorical features: {high_cardinality_cols}")
                validation_results['high_cardinality_features'] = [{"column": col, "unique_values": n} for col, n in high_cardinality_cols]
        
        # Target distribution statistics
        target_stats = self.df[self.target].describe().to_dict()
        validation_results['target_stats'] = target_stats
        
        # Check for outliers in target
        q1 = target_stats['25%']
        q3 = target_stats['75%']
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        outliers_count = ((self.df[self.target] > upper_bound) | (self.df[self.target] < lower_bound)).sum()
        outliers_pct = (outliers_count / len(self.df)) * 100
        
        validation_results['target_outliers_count'] = int(outliers_count)
        validation_results['target_outliers_pct'] = float(outliers_pct)
        
        if outliers_pct > 5:
            logger.warning(f"Target contains {outliers_count:,} outliers ({outliers_pct:.2f}% of data)")
            logger.debug(f"Target outlier bounds: lower={lower_bound:.4g}, upper={upper_bound:.4g}")
        
        validation_time = (datetime.now() - validation_start_time).total_seconds()
        logger.info(f"Data validation complete in {validation_time:.2f} seconds")
        logger.info(f"Target column '{self.target}' summary: Min={target_stats['min']:.4g}, Max={target_stats['max']:.4g}, Mean={target_stats['mean']:.4g}, Std={target_stats['std']:.4g}")
        
        return True
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess the data for regression with optional custom feature engineering.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Preprocessing data for regression...")
        preprocess_start_time = datetime.now()
        
        # Log start of preprocessing
        logger.debug(f"DataFrame shape before preprocessing: {self.df.shape[0]:,} rows, {self.df.shape[1]:,} columns")
        
        # Call the parent class method with custom feature engineering if provided
        X_train, X_test, y_train, y_test = super().preprocess_data()
        
        # Log additional regression-specific preprocessing information
        logger.debug(f"Train set: {X_train.shape[0]:,} rows ({X_train.shape[0]/self.df.shape[0]*100:.1f}% of data)")
        logger.debug(f"Test set: {X_test.shape[0]:,} rows ({X_test.shape[0]/self.df.shape[0]*100:.1f}% of data)")
        
        # Log target distribution in train and test sets
        train_target_stats = pd.Series(y_train).describe().to_dict()
        test_target_stats = pd.Series(y_test).describe().to_dict()
        
        logger.debug("Target distribution in train set:")
        logger.debug(f"  Min: {train_target_stats['min']:.4g}, Max: {train_target_stats['max']:.4g}")
        logger.debug(f"  Mean: {train_target_stats['mean']:.4g}, Std: {train_target_stats['std']:.4g}")
        logger.debug(f"  25%: {train_target_stats['25%']:.4g}, 50%: {train_target_stats['50%']:.4g}, 75%: {train_target_stats['75%']:.4g}")
        
        logger.debug("Target distribution in test set:")
        logger.debug(f"  Min: {test_target_stats['min']:.4g}, Max: {test_target_stats['max']:.4g}")
        logger.debug(f"  Mean: {test_target_stats['mean']:.4g}, Std: {test_target_stats['std']:.4g}")
        logger.debug(f"  25%: {test_target_stats['25%']:.4g}, 50%: {test_target_stats['50%']:.4g}, 75%: {test_target_stats['75%']:.4g}")
        
        # Check if distributions are similar
        mean_diff_pct = abs(train_target_stats['mean'] - test_target_stats['mean']) / train_target_stats['mean'] * 100
        if mean_diff_pct > 10:  # Arbitrary threshold
            logger.warning(f"Train and test set means differ by {mean_diff_pct:.2f}%, which may indicate data leakage or poor splitting")
        
        # Check for feature correlation with target
        try:
            # Try to get feature names from preprocessor if available
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    logger.debug(f"Preprocessor outputs {len(feature_names)} features")
                except:
                    logger.debug("Could not get feature names from preprocessor")
        except Exception as e:
            logger.debug(f"Error checking preprocessor features: {str(e)}")
        
        preprocess_time = (datetime.now() - preprocess_start_time).total_seconds()
        logger.info(f"Data preprocessing for regression completed in {preprocess_time:.2f} seconds")
        
        # Add more details to metadata
        if 'preprocessing' not in self.metadata:
            self.metadata['preprocessing'] = {}
        
        self.metadata['preprocessing']['regression_specific'] = {
            'train_target_stats': train_target_stats,
            'test_target_stats': test_target_stats,
            'target_mean_diff_pct': float(mean_diff_pct)
        }
        
        return X_train, X_test, y_train, y_test

    def train_models(self) -> Dict[str, Any]:
        """
        Train regression models, either custom or standard models.
        With optional Random Search optimization for hyperparameters.
        
        Returns:
            dict: Trained models
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        logger.info("Training regression models...")
        train_start_time = datetime.now()
        
        if self.preprocessor is None:
            error_msg = "Data not preprocessed. Call preprocess_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log training dimensions
        logger.debug(f"Training data dimensions - X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        
        training_metadata = {'models': {}}
        
        # Process training data once
        logger.info("Preprocessing training data...")
        preprocess_start = datetime.now()
        X_train_processed = self.preprocessor.transform(self.X_train)
        preprocess_time = (datetime.now() - preprocess_start).total_seconds()
        logger.debug(f"Training data preprocessing completed in {preprocess_time:.2f} seconds")
        logger.debug(f"Processed training data shape: {X_train_processed.shape}")
        
        # Check for sparse data
        is_sparse = hasattr(X_train_processed, 'toarray') and callable(getattr(X_train_processed, 'toarray'))
        if is_sparse:
            sparsity = 1.0 - (X_train_processed.nnz / (X_train_processed.shape[0] * X_train_processed.shape[1]))
            logger.debug(f"Processed data is sparse with sparsity {sparsity:.2%}")
        
        # Get model configurations from config
        model_params = self.model_config.get('models', {}).get('parameters', {})
        enabled_models = self.model_config.get('models', {}).get('enabled', [])
        
        logger.debug(f"Model parameters from config: {model_params}")
        logger.debug(f"Enabled models: {enabled_models}")
        
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
        
        # Log model configurations
        for name, model in all_regressors.items():
            logger.debug(f"Model {name} configuration: {model.get_params()}")
        
        # Filter by enabled models if specified
        regressors = {k: v for k, v in all_regressors.items() if k in enabled_models}
        logger.info(f"Training {len(regressors)} models: {', '.join(regressors.keys())}")
        
        # Check if random search is enabled
        random_search_config = self.model_config.get('random_search', {})
        use_random_search = random_search_config.get('enabled', False)
        
        if use_random_search:
            logger.info(f"Random search optimization is enabled with {random_search_config.get('n_iter', 20)} iterations")
        else:
            logger.debug("Random search optimization is disabled")
        
        # Train all models
        self.models = {}
        
        # First, train models with regular parameters
        for name, regressor in regressors.items():
            try:
                logger.info(f"Training {name}...")
                model_metadata = {
                    'model_type': str(type(regressor)),
                    'parameters': str(regressor.get_params())
                }
                
                # Log memory before training
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / (1024 * 1024)
                    logger.debug(f"Memory before training {name}: {memory_before:.2f} MB")
                except ImportError:
                    pass
                
                # Time the training
                train_start = datetime.now()
                
                model = regressor.fit(X_train_processed, self.y_train)
                
                train_time = (datetime.now() - train_start).total_seconds()
                logger.info(f"Training {name} completed in {train_time:.2f} seconds")
                
                # Log memory after training
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_after = process.memory_info().rss / (1024 * 1024)
                    memory_diff = memory_after - memory_before
                    logger.debug(f"Memory after training {name}: {memory_after:.2f} MB (change: {memory_diff:+.2f} MB)")
                except ImportError:
                    pass
                
                # Log model details
                if hasattr(model, 'feature_importances_'):
                    top_importances = sorted(zip(range(X_train_processed.shape[1]), model.feature_importances_), 
                                           key=lambda x: x[1], reverse=True)[:10]
                    logger.debug(f"Top feature importances for {name}:")
                    for idx, importance in top_importances:
                        feature_name = f"Feature_{idx}"
                        logger.debug(f"  - {feature_name}: {importance:.4f}")
                    
                    model_metadata['feature_importances'] = {f"Feature_{i}": float(imp) for i, imp in top_importances}
                
                # Store the model
                self.models[name] = model
                
                # Update metadata
                model_metadata['training_time_seconds'] = train_time
                model_metadata['trained_successfully'] = True
                model_metadata['hyperparameter_optimization'] = False
                
                # Add model complexity metrics
                try:
                    complexity_metrics = self._get_model_complexity_metrics(model)
                    if complexity_metrics:
                        model_metadata['complexity_metrics'] = complexity_metrics
                        logger.debug(f"Model complexity metrics for {name}: {complexity_metrics}")
                except Exception as e:
                    logger.debug(f"Could not calculate complexity metrics for {name}: {str(e)}")
                
                training_metadata['models'][name] = model_metadata
                
            except Exception as e:
                error_msg = f"Failed to train {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Traceback: {traceback.format_exc()}")
                training_metadata['models'][name] = {
                    'trained_successfully': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Next, apply Random Search optimization if enabled
        if use_random_search:
            logger.info("Starting Random Search optimization...")
            n_iter = random_search_config.get('n_iter', 20)
            cv = random_search_config.get('cv', 5)
            verbose = random_search_config.get('verbose', 1)
            rs_models_config = random_search_config.get('models', {})
            
            logger.debug(f"Random Search configuration: n_iter={n_iter}, cv={cv}, verbose={verbose}")
            logger.debug(f"Models to optimize: {list(rs_models_config.keys())}")
            
            # Apply Random Search to configured models
            for model_name, model_config in rs_models_config.items():
                if model_config.get('enabled', False) and model_name in regressors:
                    logger.info(f"Performing Random Search optimization for {model_name}")
                    
                    # Get base model
                    base_model = regressors[model_name]
                    
                    # Get parameter distributions
                    param_distributions = model_config.get('parameters', {})
                    
                    # Verify param_distributions is suitable for RandomizedSearchCV
                    if not param_distributions:
                        logger.warning(f"No parameter distributions defined for {model_name}, skipping Random Search")
                        continue
                    
                    logger.debug(f"Parameter distributions for {model_name}: {param_distributions}")
                    
                    try:
                        # Create RandomizedSearchCV
                        logger.debug(f"Creating RandomizedSearchCV for {model_name}")
                        random_search = RandomizedSearchCV(
                            base_model,
                            param_distributions=param_distributions,
                            n_iter=n_iter,
                            cv=cv,
                            scoring='neg_mean_squared_error',
                            random_state=self.random_state,
                            verbose=verbose,
                            n_jobs=-1
                        )
                        
                        # Time the optimization
                        train_start = datetime.now()
                        
                        logger.info(f"Starting Random Search for {model_name} with {n_iter} iterations and {cv}-fold CV")
                        
                        # Log memory before optimization
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_before = process.memory_info().rss / (1024 * 1024)
                            logger.debug(f"Memory before Random Search for {model_name}: {memory_before:.2f} MB")
                        except ImportError:
                            pass
                        
                        # Fit on processed data
                        random_search.fit(X_train_processed, self.y_train)
                        
                        # Log memory after optimization
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_after = process.memory_info().rss / (1024 * 1024)
                            memory_diff = memory_after - memory_before
                            logger.debug(f"Memory after Random Search for {model_name}: {memory_after:.2f} MB (change: {memory_diff:+.2f} MB)")
                        except ImportError:
                            pass
                        
                        # Add optimized model with a distinct name
                        optimized_name = f"{model_name}_optimized"
                        self.models[optimized_name] = random_search.best_estimator_
                        
                        # Calculate training time
                        train_time = (datetime.now() - train_start).total_seconds()
                        logger.info(f"Random Search for {model_name} completed in {train_time:.2f} seconds")
                        logger.info(f"Best parameters: {random_search.best_params_}")
                        
                        # Log the best score
                        best_mse = -random_search.best_score_
                        best_rmse = np.sqrt(best_mse)
                        logger.info(f"Best CV score (RMSE): {best_rmse:.4f}")
                        
                        # Log all CV results
                        cv_results = pd.DataFrame(random_search.cv_results_)
                        best_index = random_search.best_index_
                        logger.debug(f"CV results summary:")
                        logger.debug(f"  - Mean test score: {cv_results['mean_test_score'].mean()}")
                        logger.debug(f"  - Std test score: {cv_results['std_test_score'].mean()}")
                        logger.debug(f"  - Min test score: {cv_results['mean_test_score'].min()}")
                        logger.debug(f"  - Max test score: {cv_results['mean_test_score'].max()}")
                        
                        # Log feature importances if available
                        if hasattr(random_search.best_estimator_, 'feature_importances_'):
                            top_importances = sorted(zip(range(X_train_processed.shape[1]), 
                                                      random_search.best_estimator_.feature_importances_), 
                                                   key=lambda x: x[1], reverse=True)[:10]
                            logger.debug(f"Top feature importances for optimized {model_name}:")
                            for idx, importance in top_importances:
                                feature_name = f"Feature_{idx}"
                                logger.debug(f"  - {feature_name}: {importance:.4f}")
                        
                        # Update metadata
                        optimized_metadata = {
                            'model_type': str(type(random_search.best_estimator_)),
                            'parameters': str(random_search.best_params_),
                            'training_time_seconds': train_time,
                            'trained_successfully': True,
                            'hyperparameter_optimization': True,
                            'optimization_method': 'random_search',
                            'n_iter': n_iter,
                            'cv_folds': cv,
                            'best_score': float(best_rmse),  # Store as RMSE for consistency
                            'cv_results_summary': {
                                'mean_score': float(-cv_results['mean_test_score'].mean()),  # Convert from neg MSE
                                'std_score': float(cv_results['std_test_score'].mean()),
                                'min_score': float(-cv_results['mean_test_score'].min()),  # Convert from neg MSE
                                'max_score': float(-cv_results['mean_test_score'].max())  # Convert from neg MSE
                            }
                        }
                        
                        # Add feature importances if available
                        if hasattr(random_search.best_estimator_, 'feature_importances_'):
                            optimized_metadata['feature_importances'] = {
                                f"Feature_{i}": float(imp) for i, imp in top_importances
                            }
                        
                        # Add model complexity metrics
                        try:
                            complexity_metrics = self._get_model_complexity_metrics(random_search.best_estimator_)
                            if complexity_metrics:
                                optimized_metadata['complexity_metrics'] = complexity_metrics
                        except Exception as e:
                            logger.debug(f"Could not calculate complexity metrics for optimized {model_name}: {str(e)}")
                        
                        training_metadata['models'][optimized_name] = optimized_metadata
                        
                    except Exception as e:
                        error_msg = f"Failed to optimize {model_name} with Random Search: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        training_metadata['models'][f"{model_name}_optimized"] = {
                            'trained_successfully': False,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }
        
        total_models = len(self.models)
        total_training_time = (datetime.now() - train_start_time).total_seconds()
        logger.info(f"Successfully trained {total_models} models in {total_training_time:.2f} seconds")
        
        # Log some overall performance stats
        successfully_trained = sum(1 for name, meta in training_metadata['models'].items() 
                                   if meta.get('trained_successfully', False))
        logger.info(f"Training summary: {successfully_trained}/{len(training_metadata['models'])} models trained successfully")
        
        # Update metadata
        training_metadata['total_training_time_seconds'] = total_training_time
        training_metadata['models_trained'] = total_models
        training_metadata['models_successful'] = successfully_trained
        self.metadata['models'] = training_metadata
        
        # Log memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            logger.debug(f"Memory usage after training all models: {memory_mb:.2f} MB")
        except ImportError:
            pass
        
        return self.models
    
    def _get_model_complexity_metrics(self, model):
        """Get complexity metrics for a model."""
        metrics = {}
        
        # For tree-based models
        if hasattr(model, 'get_n_leaves'):
            metrics['n_leaves'] = model.get_n_leaves()
            
        if hasattr(model, 'get_depth'):
            metrics['depth'] = model.get_depth()
            
        # For ensemble models
        if hasattr(model, 'n_estimators'):
            metrics['n_estimators'] = model.n_estimators
            
        # For linear models
        if hasattr(model, 'coef_'):
            metrics['n_coefficients'] = model.coef_.size
            metrics['n_nonzero_coefficients'] = np.count_nonzero(model.coef_)
            
        return metrics

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
        eval_start_time = datetime.now()
        
        # Log evaluation process
        logger.debug(f"Evaluating {len(self.models)} trained models")
        logger.debug(f"Test data dimensions - X_test: {self.X_test.shape}, y_test: {self.y_test.shape}")
        
        evaluation_metadata = {'models': {}}
        
        # Get evaluation configuration
        eval_metrics = self.model_config.get('evaluation', {}).get('metrics', ['r2', 'rmse', 'mae'])
        primary_metric = self.model_config.get('evaluation', {}).get('primary_metric', 'r2')
        
        logger.debug(f"Evaluation metrics: {eval_metrics}, primary metric: {primary_metric}")
        
        # Transform test data
        preproc_start = datetime.now()
        X_test_processed = self.preprocessor.transform(self.X_test)
        preproc_time = (datetime.now() - preproc_start).total_seconds()
        logger.debug(f"Test data preprocessing completed in {preproc_time:.2f} seconds")
        
        # Check if primary metric is minimizing or maximizing
        if primary_metric in ['rmse', 'mae', 'mape']:
            is_minimizing = True
            logger.debug(f"Primary metric '{primary_metric}' should be minimized")
        else:
            is_minimizing = False
            logger.debug(f"Primary metric '{primary_metric}' should be maximized")
        
        results = []
        
        # Store primary metric for each model
        all_models_primary_metric = {}
        
        # Track performance ranges for reporting
        metric_ranges = {
            'train_r2': {'min': float('inf'), 'max': float('-inf')},
            'test_r2': {'min': float('inf'), 'max': float('-inf')},
            'train_rmse': {'min': float('inf'), 'max': float('-inf')},
            'test_rmse': {'min': float('inf'), 'max': float('-inf')},
            'train_mae': {'min': float('inf'), 'max': float('-inf')},
            'test_mae': {'min': float('inf'), 'max': float('-inf')},
            'train_mape': {'min': float('inf'), 'max': float('-inf')},
            'test_mape': {'min': float('inf'), 'max': float('-inf')}
        }

        for name, model in self.models.items():
            try:
                logger.info(f"Evaluating {name}...")
                model_eval_metadata = {}
                eval_start = datetime.now()
                
                # Make predictions
                train_pred_start = datetime.now()
                y_train_pred = model.predict(self.preprocessor.transform(self.X_train))
                train_pred_time = (datetime.now() - train_pred_start).total_seconds()
                
                test_pred_start = datetime.now()
                y_test_pred = model.predict(X_test_processed)
                test_pred_time = (datetime.now() - test_pred_start).total_seconds()
                
                logger.debug(f"Prediction times - Train: {train_pred_time:.2f}s, Test: {test_pred_time:.2f}s")
                
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
                
                # Update metric ranges
                for metric_name, value in metrics.items():
                    if metric_name != 'model' and not pd.isna(value):
                        if value < metric_ranges[metric_name]['min']:
                            metric_ranges[metric_name]['min'] = value
                        if value > metric_ranges[metric_name]['max']:
                            metric_ranges[metric_name]['max'] = value
                
                # Calculate prediction error statistics
                test_errors = self.y_test - y_test_pred
                error_stats = {
                    'mean_error': float(np.mean(test_errors)),
                    'median_error': float(np.median(test_errors)),
                    'std_error': float(np.std(test_errors)),
                    'abs_mean_error': float(np.mean(np.abs(test_errors))),
                    'abs_median_error': float(np.median(np.abs(test_errors))),
                    'min_error': float(np.min(test_errors)),
                    'max_error': float(np.max(test_errors)),
                    'q1_error': float(np.percentile(test_errors, 25)),
                    'q3_error': float(np.percentile(test_errors, 75))
                }
                
                # Check for test performance drop
                r2_drop = metrics['train_r2'] - metrics['test_r2']
                rmse_increase = metrics['test_rmse'] - metrics['train_rmse']
                
                # Log more detailed analysis
                overfitting_detected = False
                if r2_drop > 0.2:  # Arbitrary threshold
                    logger.warning(f"{name}: Possible overfitting - R² drop of {r2_drop:.4f} (train: {metrics['train_r2']:.4f}, test: {metrics['test_r2']:.4f})")
                    overfitting_detected = True
                
                if rmse_increase / metrics['train_rmse'] > 0.3:  # Arbitrary threshold (30% increase)
                    logger.warning(f"{name}: Possible overfitting - RMSE increase of {rmse_increase:.4f} ({(rmse_increase/metrics['train_rmse'])*100:.1f}%)")
                    overfitting_detected = True
                
                # Check for extreme prediction errors
                error_threshold = 3 * np.std(test_errors)
                extreme_errors = np.abs(test_errors) > error_threshold
                extreme_error_count = np.sum(extreme_errors)
                
                if extreme_error_count > 0:
                    extreme_error_pct = extreme_error_count / len(test_errors) * 100
                    logger.debug(f"{name}: Found {extreme_error_count} extreme prediction errors ({extreme_error_pct:.2f}% of test data)")
                    
                    # Sample some of these extreme errors for analysis
                    if extreme_error_count > 0:
                        extreme_indices = np.where(extreme_errors)[0]
                        sample_size = min(3, len(extreme_indices))
                        sample_indices = np.random.choice(extreme_indices, sample_size, replace=False)
                        
                        logger.debug("Sample of extreme errors:")
                        for idx in sample_indices:
                            actual = self.y_test.iloc[idx] if hasattr(self.y_test, 'iloc') else self.y_test[idx]
                            predicted = y_test_pred[idx]
                            error = actual - predicted
                            logger.debug(f"  - Index {idx}: Actual={actual:.4g}, Predicted={predicted:.4g}, Error={error:.4g}")
                
                # Save primary metric value for this model
                metric_col = f'test_{primary_metric}'
                primary_metric_value = metrics[metric_col]
                all_models_primary_metric[name] = float(primary_metric_value)
                
                # Calculate prediction time per instance
                if hasattr(self.X_test, 'shape'):
                    pred_time_per_instance = test_pred_time / self.X_test.shape[0]
                    metrics['prediction_time_ms'] = pred_time_per_instance * 1000  # Convert to ms
                
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
                
                if 'prediction_time_ms' in metrics:
                    model_eval_metadata['metrics']['prediction_time_ms'] = float(metrics['prediction_time_ms'])
                
                model_eval_metadata['error_analysis'] = error_stats
                model_eval_metadata['extreme_errors'] = {
                    'count': int(extreme_error_count),
                    'percentage': float(extreme_error_pct) if 'extreme_error_pct' in locals() else 0.0,
                    'threshold': float(error_threshold)
                }
                
                model_eval_metadata['overfitting_analysis'] = {
                    'r2_drop': float(r2_drop),
                    'rmse_increase': float(rmse_increase),
                    'rmse_increase_pct': float(rmse_increase / metrics['train_rmse'] * 100),
                    'overfitting_detected': overfitting_detected
                }
                
                eval_time = (datetime.now() - eval_start).total_seconds()
                model_eval_metadata['evaluation_time_seconds'] = eval_time
                
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
                logger.debug(f"Traceback: {traceback.format_exc()}")
                evaluation_metadata['models'][name] = {
                    'evaluation_error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Log metric ranges
        logger.debug("Metric ranges across all models:")
        for metric_name, range_values in metric_ranges.items():
            if range_values['min'] != float('inf') and range_values['max'] != float('-inf'):
                logger.debug(f"  {metric_name}: min={range_values['min']:.4f}, max={range_values['max']:.4f}, range={range_values['max']-range_values['min']:.4f}")
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        evaluation_metadata['results_summary'] = self.results.to_dict()
        evaluation_metadata['metric_ranges'] = {m: {k: float(v) for k, v in r.items()} 
                                                for m, r in metric_ranges.items() 
                                                if r['min'] != float('inf') and r['max'] != float('-inf')}
        
        # Store all models primary metric in metadata
        evaluation_metadata['primary_metric'] = primary_metric
        evaluation_metadata['all_models_primary_metric'] = all_models_primary_metric
        evaluation_metadata['is_minimizing_metric'] = is_minimizing
        
        # Identify best model based on primary metric
        if not self.results.empty:
            metric_col = f'test_{primary_metric}'
            
            if primary_metric in ['rmse', 'mae', 'mape']:  # These are error metrics, lower is better
                best_idx = self.results[metric_col].idxmin()
                best_comp = "lowest"
            else:  # For R², higher is better
                best_idx = self.results[metric_col].idxmax()
                best_comp = "highest"
                
            best_model_name = self.results.loc[best_idx, 'model']
            best_metric_value = self.results.loc[best_idx, metric_col]
            
            # Log performance improvement over baseline
            if 'linear_regression' in self.models:
                baseline_model = 'linear_regression'
                baseline_idx = self.results[self.results['model'] == baseline_model].index
                if len(baseline_idx) > 0:
                    baseline_idx = baseline_idx[0]
                    baseline_value = self.results.loc[baseline_idx, metric_col]
                    
                    if is_minimizing:
                        improvement = (baseline_value - best_metric_value) / baseline_value * 100
                        logger.info(f"Best model improves over linear regression baseline by {improvement:.2f}% ({best_metric_value:.4f} vs {baseline_value:.4f})")
                    else:
                        improvement = (best_metric_value - baseline_value) / baseline_value * 100
                        logger.info(f"Best model improves over linear regression baseline by {improvement:.2f}% ({best_metric_value:.4f} vs {baseline_value:.4f})")
                    
                    evaluation_metadata['improvement_over_baseline'] = {
                        'baseline_model': baseline_model,
                        'baseline_value': float(baseline_value),
                        'best_value': float(best_metric_value),
                        'improvement_pct': float(improvement)
                    }
            
            self.best_model = self.models[best_model_name]
            logger.info(f"\nBest model: {best_model_name} (Test {primary_metric.upper()} = {best_metric_value:.4f}, {best_comp} among all models)")
            
            # Get predictions from the best model
            best_pred_start = datetime.now()
            best_test_predictions = self.best_model.predict(X_test_processed)
            best_pred_time = (datetime.now() - best_pred_start).total_seconds()
            
            logger.debug(f"Best model prediction time: {best_pred_time:.4f} seconds for {len(self.y_test)} instances")
            
            # Additional analysis of best model predictions
            best_errors = self.y_test - best_test_predictions
            abs_errors = np.abs(best_errors)
            
            # Error distribution by percentile
            error_percentiles = [10, 25, 50, 75, 90, 95, 99]
            error_by_percentile = {f"p{p}": float(np.percentile(abs_errors, p)) for p in error_percentiles}
            
            logger.debug(f"Best model absolute error distribution:")
            for p, v in error_by_percentile.items():
                logger.debug(f"  {p}: {v:.4f}")
            
            # Error by target value range
            if hasattr(self.y_test, 'values'):
                y_values = self.y_test.values
            else:
                y_values = self.y_test
                
            # Analyze errors by target value quantiles
            try:
                target_quantiles = 4  # Quartiles
                y_test_quantiles = pd.qcut(y_values, target_quantiles, duplicates='drop')
                quantile_labels = sorted(set(y_test_quantiles))
                
                logger.debug(f"Error analysis by target value range:")
                for q in quantile_labels:
                    mask = (y_test_quantiles == q)
                    q_errors = abs_errors[mask]
                    q_mae = np.mean(q_errors)
                    q_mape = np.mean(q_errors / np.abs(y_values[mask])) * 100
                    logger.debug(f"  Range {q}: MAE={q_mae:.4f}, MAPE={q_mape:.2f}%, Count={np.sum(mask)}")
            except Exception as e:
                logger.debug(f"Could not analyze errors by target range: {str(e)}")

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
            
            # Add error distribution analysis
            self.metadata['best_model']['error_analysis'] = {
                'percentiles': error_by_percentile,
                'mean_abs_error': float(np.mean(abs_errors)),
                'median_abs_error': float(np.median(abs_errors)),
                'std_error': float(np.std(best_errors)),
                'mean_error': float(np.mean(best_errors)),  # Bias
                'max_abs_error': float(np.max(abs_errors))
            }
        
        eval_total_time = (datetime.now() - eval_start_time).total_seconds()
        logger.info(f"Model evaluation completed in {eval_total_time:.2f} seconds")
        
        # Update overall metadata
        evaluation_metadata['evaluation_time_seconds'] = eval_total_time
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
        store_start = datetime.now()
        
        # Create a DataFrame with actual and predicted values
        prediction_df = pd.DataFrame({
            'actual': self.y_test.values if hasattr(self.y_test, 'values') else self.y_test,
            'predicted': predictions
        })
            
        # Fallback to simpler sampling
        if len(prediction_df) > max_predictions:
            sampled_predictions = prediction_df.sample(n=max_predictions, random_state=42)
            logger.info(f"Storing {len(sampled_predictions)} predictions in metadata (randomly sampled from {len(prediction_df)} total)")
        else:
            # Store all predictions if under the limit
            sampled_predictions = prediction_df
            logger.info(f"Storing all {len(prediction_df)} predictions in metadata")
    
        # Convert to list for storing in metadata
        predictions_list = []
        for idx, row in sampled_predictions.iterrows():
            pred_entry = {
                'actual': float(row['actual']),
                'predicted': float(row['predicted'])
            }
                
            predictions_list.append(pred_entry)
        
        # Add to metadata
        self.metadata['best_model']['prediction_samples'] = predictions_list
        

        
        store_time = (datetime.now() - store_start).total_seconds()
        logger.debug(f"Stored prediction samples in {store_time:.2f} seconds")