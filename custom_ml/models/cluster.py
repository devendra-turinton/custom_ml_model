from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional, Tuple
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from custom_ml.src import ml_utils
from custom_ml.training_pipeline import BasePipeline
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)


class ClusteringPipeline(BasePipeline):
    """
    Pipeline for clustering tasks.
    
    This pipeline is designed for unsupervised clustering of data.
    It supports multiple clustering algorithms and methods for estimating 
    the optimal number of clusters.
    """
    
    def __init__(
            self, 
            data_path: Optional[str] = None,
            df: Optional[pd.DataFrame] = None,
            n_clusters: Optional[int] = None,
            config_path: Optional[str] = "config/config.yaml",
            output_dir: Optional[str] = None,
            model_id: Optional[str] = None,
            version: Optional[str] = None,
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
            output_dir: Output directory for model artifacts
            model_id: Unique identifier for the model
            version: Version string
            cluster_range: Range of clusters to try for optimal cluster selection
            dim_reduction: Dimensionality reduction method ('pca', 'tsne', 'none')
            n_components: Number of components for dimensionality reduction
        """
        # Call the parent class constructor without a target variable
        super().__init__(data_path=data_path, df=df, target=None, 
                         config_path=config_path, output_dir=output_dir, 
                         model_id=model_id, version=version)
        
        # Initialize clustering-specific attributes
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
        
        # Update metadata with clustering parameters
        self.metadata['parameters'].update({
            'problem_type': 'clustering',
            'n_clusters': n_clusters,
            'cluster_range': [int(cluster_range[0]), int(cluster_range[1])],
            'dim_reduction': dim_reduction,
            'n_components': n_components
        })
        
        logger.info(f"Clustering Pipeline initialized")
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
        validation_start_time = datetime.now()
        
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
            logger.debug(f"Found {len(numeric_columns)} numeric columns")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if not columns_with_missing.empty:
            missing_data = {}
            logger.warning(f"Found {len(columns_with_missing)} columns with missing values")
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
            logger.info("No missing values found in the dataset")
        
        # Check sample size
        if len(self.df) < 10:
            logger.warning(f"Very small dataset with only {len(self.df)} samples. Clustering may not be effective.")
        
        # Check dimensionality
        if len(self.df.columns) > 100:
            logger.warning(f"High-dimensional data with {len(self.df.columns)} features. Consider dimensionality reduction.")
        
        # Update metadata with validation results
        self.metadata['data']['validation'] = validation_results
        
        validation_time = (datetime.now() - validation_start_time).total_seconds()
        logger.info(f"Data validation completed in {validation_time:.2f} seconds")
        logger.info(f"Data validated: {len(self.df.columns)} feature columns available for clustering")
        
        return True
    
    def preprocess_data(self) -> np.ndarray:
        """
        Preprocess the data for clustering.
        
        Returns:
            np.ndarray: Preprocessed data ready for clustering
        """
        logger.info("Preprocessing data for clustering...")
        preprocess_start_time = datetime.now()
        
        preprocessing_metadata = {}
        
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
        
        logger.debug(f"Data shape after preprocessing: {self.X_transformed.shape}")
        
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
        
        logger.debug(f"Data shape after scaling: {self.X_scaled.shape}")
        
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
        
        preprocessing_time = (datetime.now() - preprocess_start_time).total_seconds()
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
        logger.info(f"Applying {self.dim_reduction} dimensionality reduction to {self.n_components} components")
        dim_reduction_start_time = datetime.now()
        
        dim_reduction_metadata = {}
        
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
                    
                    # Store explained variance in metadata
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
            logger.info("TSNE can be slow for large datasets. Starting dimensionality reduction...")
            self.X_reduced = reducer.fit_transform(self.X_scaled)
            logger.info("TSNE dimensionality reduction completed")
        else:
            logger.warning(f"Unknown dimensionality reduction method: {self.dim_reduction}")
            self.X_reduced = self.X_scaled
        
        dim_reduction_time = (datetime.now() - dim_reduction_start_time).total_seconds()
        logger.info(f"Dimensionality reduction completed in {dim_reduction_time:.2f} seconds")
        
        dim_reduction_metadata['time_seconds'] = dim_reduction_time
        dim_reduction_metadata['output_shape'] = self.X_reduced.shape
        
        # Update metadata
        if 'preprocessing' not in self.metadata:
            self.metadata['preprocessing'] = {}
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
        logger.info("Estimating optimal number of clusters...")
        estimation_start_time = datetime.now()
        
        if self.n_clusters is not None:
            logger.info(f"Using user-specified number of clusters: {self.n_clusters}")
            self.optimal_clusters = self.n_clusters
            self.metadata['models']['cluster_estimation'] = {
                'method': 'user_specified',
                'optimal_clusters': self.n_clusters
            }
            return self.n_clusters
        
        if 'models' not in self.metadata:
            self.metadata['models'] = {}
        
        estimation_metadata = {}
        
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
            logger.debug(f"Testing k={k} clusters")
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
        
        total_time = (datetime.now() - estimation_start_time).total_seconds()
        estimation_metadata['time_seconds'] = total_time
        
        # Update metadata
        self.metadata['models']['cluster_estimation'] = estimation_metadata
        
        logger.info(f"Selected optimal number of clusters: {self.optimal_clusters}")
        return self.optimal_clusters
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train multiple clustering models.
        
        Returns:
            dict: Trained models with their labels
        """
        logger.info("Training clustering models...")
        train_start_time = datetime.now()
        
        # Use the preprocessed data
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
        
        # Initialize models dictionary and metadata
        self.models = {}
        if 'models' not in self.metadata:
            self.metadata['models'] = {}
        
        training_metadata = {'models': {}}
        training_metadata['n_clusters'] = n_clusters
        
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
        
        total_time = (datetime.now() - train_start_time).total_seconds()
        logger.info(f"All clustering models trained in {total_time:.2f} seconds")
        
        # Update metadata
        training_metadata['total_time_seconds'] = total_time
        training_metadata['n_models_trained'] = len(self.models)
        self.metadata['models']['training'] = training_metadata
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate trained clustering models.
        
        Returns:
            pd.DataFrame: Results with performance metrics
        """
        if not self.models:
            error_msg = "No trained models. Call train_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Evaluating clustering models...")
        eval_start_time = datetime.now()
        
        # Use the preprocessed data
        if self.X_reduced is not None:
            data = self.X_reduced
        else:
            data = self.X_scaled
        
        evaluation_metadata = {}
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
                logger.info(f"Best model based on silhouette score: {best_model_name} (Silhouette = {best_silhouette:.4f})")
                
                # Initialize best_model metadata if not exists
                if 'best_model' not in self.metadata:
                    self.metadata['best_model'] = {}
                
                # Update best_model metadata
                self.metadata['best_model'].update({
                    'name': best_model_name,
                    'model_type': str(type(self.best_model['model']).__name__),
                    'primary_metric': 'silhouette',
                    'primary_metric_value': float(best_silhouette),
                    'n_clusters': int(non_nan_results.loc[best_idx, 'n_clusters'])
                })
                
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
                    logger.info(f"Best model based on cluster count: {best_model_name}")
                    
                    # Initialize best_model metadata if not exists
                    if 'best_model' not in self.metadata:
                        self.metadata['best_model'] = {}
                    
                    # Update best_model metadata
                    self.metadata['best_model'].update({
                        'name': best_model_name,
                        'model_type': str(type(self.best_model['model']).__name__),
                        'primary_metric': 'cluster_count',
                        'criterion': 'closest_to_target_clusters',
                        'n_clusters': int(self.results.loc[best_idx, 'n_clusters'])
                    })
                    
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
                    
                    # Initialize best_model metadata if not exists
                    if 'best_model' not in self.metadata:
                        self.metadata['best_model'] = {}
                    
                    # Update best_model metadata
                    self.metadata['best_model'].update({
                        'name': best_model_name,
                        'model_type': str(type(self.best_model['model']).__name__),
                        'primary_metric': 'default_selection',
                        'criterion': 'default_first_model'
                    })
                    
                    evaluation_metadata['best_model'] = {
                        'name': best_model_name,
                        'criterion': 'default_first_model'
                    }
        
        total_time = (datetime.now() - eval_start_time).total_seconds()
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
            'problem_type': 'clustering',
            'timestamp': datetime.now().isoformat()
        }
        
        model_filename = os.path.join(self.output_dir, f"{self.model_id}.pkl")
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(model_package, f)
        
        save_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Best model ({best_model_name}) saved to {model_filename} in {save_time:.2f} seconds")
        
        # Update metadata
        self.metadata['best_model']['filename'] = model_filename
        self.metadata['best_model']['save_time_seconds'] = save_time
        
        return model_filename
    
    def run_pipeline(self) -> Tuple[Any, pd.DataFrame]:
        """
        Run the complete pipeline.
        
        Returns:
            tuple: (best_model, evaluation_results)
        """
        logger.info(f"Starting Clustering ML pipeline run - ID: {self.model_id}")
        pipeline_start = datetime.now()
        
        try:
            # Load and validate data
            logger.info("\n" + "="*50)
            logger.info("STEP 1: Loading and validating data")
            logger.info("="*50)
            step_start = datetime.now()
            self.load_data()
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
            logger.info(f"Data preprocessed and ready for clustering")
            logger.info(f"Step completed in {step_time:.2f} seconds")
            
            # Train models
            logger.info("\n" + "="*50)
            logger.info("STEP 3: Training models")
            logger.info("="*50)
            step_start = datetime.now()
            self.train_models()
            step_time = (datetime.now() - step_start).total_seconds()
            logger.info(f"Trained {len(self.models):,} clustering models")
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
            
            # Update metadata with error information
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['failure_timestamp'] = datetime.now().isoformat()
            self.save_metadata()
            
            # Re-raise the exception
            raise