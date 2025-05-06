from datetime import datetime
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src import ml_utils
from training_pipeline import BasePipeline
logger = logging.getLogger(__name__)

# For Time Series pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error



# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import logging
logger = logging.getLogger(__name__)



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
        super().__init__(**kwargs)
        self.problem_type = 'regression'
        
        # Get regression-specific configuration
        self.model_config = ml_utils.get_model_config(self.config, 'regression')
        
        logger.info("Regression Pipeline initialized")
    
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
        return super().preprocess_data()
    
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
        regressors = {k: v for k, v in all_regressors.items() if k in enabled_models}
        logger.info(f"Training {len(regressors)} models: {', '.join(regressors.keys())}")
    
        
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

