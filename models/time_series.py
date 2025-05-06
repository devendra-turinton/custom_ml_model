
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Optional

from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src import ml_utils
from training_pipeline import BasePipeline
logger = logging.getLogger(__name__)

# For Time Series pipeline
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error



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
        feature_extractor = ml_utils.TimeSeriesFeatureExtractor(
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