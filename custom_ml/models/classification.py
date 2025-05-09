from datetime import datetime
import numpy as np
import pandas as pd
import logging
import os
import traceback
from typing import Any, Dict, Optional, Tuple

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from custom_ml.src import ml_utils
from custom_ml.training_pipeline import BasePipeline
logger = logging.getLogger(__name__)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    balanced_accuracy_score, log_loss, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)

logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
#########################################
# Classification Pipeline
#########################################

class ClassificationPipeline(BasePipeline):
  
    def __init__(
            self, 
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
        logger.info("Initializing Classification Pipeline...")
        init_start_time = datetime.now()
        
        # Log initialization parameters
        params_to_log = {k: v for k, v in kwargs.items() if k != 'df'}
        params_to_log.update({
            'balance_method': balance_method,
            'multi_class': multi_class,
            'eval_metric': eval_metric
        })
        logger.debug(f"Initialization parameters: {params_to_log}")
        
        super().__init__(**kwargs)
        self.problem_type = 'classification'
        self.balance_method = balance_method
        self.multi_class = multi_class
        self.eval_metric = eval_metric
        
        # Get classification-specific configuration
        config_start_time = datetime.now()
        self.model_config = ml_utils.get_model_config(self.config, 'classification')
        config_time = (datetime.now() - config_start_time).total_seconds()
        
        # Log configuration details
        if self.model_config:
            enabled_models = self.model_config.get('models', {}).get('enabled', [])
            logger.debug(f"Loaded classification configuration in {config_time:.2f} seconds")
            logger.debug(f"Enabled models: {enabled_models}")
            
            # Log balance method info
            if self.balance_method:
                logger.debug(f"Class balancing method: {self.balance_method}")
            else:
                logger.debug("No class balancing method specified")
            
            # Log evaluation metrics
            eval_config = self.model_config.get('evaluation', {})
            primary_metric = self.eval_metric
            logger.debug(f"Primary evaluation metric: {primary_metric}")
            
            # Log random search configuration if it exists
            if 'random_search' in self.model_config and self.model_config['random_search'].get('enabled', False):
                rs_config = self.model_config['random_search']
                logger.debug(f"Random search enabled: {rs_config.get('n_iter', 20)} iterations, {rs_config.get('cv', 5)}-fold CV")
                rs_models = [k for k, v in rs_config.get('models', {}).items() if v.get('enabled', False)]
                logger.debug(f"Models for random search optimization: {rs_models}")
        else:
            logger.warning("No classification-specific configuration found, using defaults")
        
        # Initialize classification-specific attributes
        self.le = None  # Label encoder for target
        self.class_names = None  # Original class names
        self.is_binary = None  # Flag for binary classification
        self.class_distribution = None  # Store class distribution
        
        # Log DataFrame information if provided
        if 'df' in kwargs and kwargs['df'] is not None:
            df = kwargs['df']
            logger.debug(f"DataFrame provided: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            
            # Check data types
            dtypes = df.dtypes.value_counts().to_dict()
            logger.debug(f"DataFrame column types: {dtypes}")
            
            # Log target column info if provided
            if 'target' in kwargs and kwargs['target'] is not None:
                target = kwargs['target']
                if target in df.columns:
                    # Get preliminary class info
                    target_values = df[target].value_counts()
                    target_unique = len(target_values)
                    logger.debug(f"Target column '{target}' has {target_unique} unique values")
                    
                    # Check if likely binary or multiclass
                    is_likely_binary = target_unique == 2
                    if is_likely_binary:
                        logger.debug(f"Target appears to be binary: values={target_values.index.tolist()}")
                    else:
                        logger.debug(f"Target appears to be multiclass with {target_unique} classes")
                        
                    # Check data type and distribution
                    logger.debug(f"Target column type: {df[target].dtype}")
                    
                    # For small number of classes, log distribution
                    if target_unique <= 10:
                        dist_pct = (target_values / len(df) * 100).round(2)
                        for cls, (count, pct) in zip(target_values.index, zip(target_values, dist_pct)):
                            logger.debug(f"  - Class '{cls}': {count:,} ({pct:.1f}%)")
                else:
                    logger.warning(f"Target column '{target}' not found in DataFrame columns")
        
        # Log system resources if psutil is available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            logger.debug(f"Memory usage after initialization: {memory_mb:.2f} MB")
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        
        init_time = (datetime.now() - init_start_time).total_seconds()        
        logger.info(f"Classification Pipeline initialized in {init_time:.2f} seconds")
            
    def validate_data(self) -> bool:
        """
        Validate the data for classification tasks.
        
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating data for classification...")
        validation_start_time = datetime.now()
        
        if self.df is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.target is None:
            error_msg = "Target column name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log data dimensions
        logger.debug(f"Data dimensions: {self.df.shape[0]:,} rows, {self.df.shape[1]:,} columns")
        
        # Validation checks
        validation_results = {}
        
        # Check if target column exists
        logger.debug(f"Checking if target column '{self.target}' exists...")
        if self.target not in self.df.columns:
            error_msg = f"Target column '{self.target}' not found in data"
            logger.error(error_msg)
            logger.debug(f"Available columns: {list(self.df.columns)}")
            raise ValueError(error_msg)
        
        validation_results['target_exists'] = True
        logger.debug(f"Target column '{self.target}' exists ✓")
        
        # Check if dataset is empty
        logger.debug("Checking if dataset is empty...")
        if len(self.df) == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['dataset_not_empty'] = True
        logger.debug(f"Dataset is not empty, contains {len(self.df):,} rows ✓")
        
        # Check target data type
        target_dtype = self.df[self.target].dtype
        logger.debug(f"Target column data type: {target_dtype}")
        validation_results['target_dtype'] = str(target_dtype)
        
        # Check for missing values in target
        missing_in_target = self.df[self.target].isna().sum()
        missing_pct = missing_in_target / len(self.df) * 100
        validation_results['missing_in_target'] = int(missing_in_target)
        validation_results['missing_in_target_pct'] = float(missing_pct)
        
        if missing_in_target > 0:
            logger.warning(f"Target column contains {missing_in_target:,} missing values ({missing_pct:.2f}%)")
        else:
            logger.debug("Target column has no missing values ✓")
        
        # Check target class distribution
        logger.debug("Analyzing target class distribution...")
        self.class_distribution = self.df[self.target].value_counts()
        validation_results['class_distribution'] = self.class_distribution.to_dict()
        
        if len(self.class_distribution) < 2:
            error_msg = f"Target column '{self.target}' must have at least 2 classes, found {len(self.class_distribution)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validation_results['has_multiple_classes'] = True
        logger.debug(f"Target has {len(self.class_distribution)} unique classes ✓")
        
        # Identify binary or multiclass problem
        self.is_binary = len(self.class_distribution) == 2
        self.problem_subtype = 'binary' if self.is_binary else 'multiclass'
        validation_results['problem_subtype'] = self.problem_subtype
        logger.debug(f"Problem identified as {self.problem_subtype} classification ✓")
        
        # Check for class imbalance
        min_class_count = self.class_distribution.min()
        max_class_count = self.class_distribution.max()
        min_class_name = self.class_distribution.idxmin()
        max_class_name = self.class_distribution.idxmax()
        
        min_class_pct = min_class_count / len(self.df) * 100
        max_class_pct = max_class_count / len(self.df) * 100
        
        imbalance_ratio = max_class_count / min_class_count
        validation_results['imbalance_ratio'] = float(imbalance_ratio)
        validation_results['min_class'] = {
            'name': str(min_class_name),
            'count': int(min_class_count),
            'percentage': float(min_class_pct)
        }
        validation_results['max_class'] = {
            'name': str(max_class_name),
            'count': int(max_class_count),
            'percentage': float(max_class_pct)
        }
        
        if imbalance_ratio > 3:
            logger.warning(f"Class imbalance detected: ratio {imbalance_ratio:.2f}")
            logger.warning(f"Smallest class '{min_class_name}': {min_class_count:,} samples ({min_class_pct:.2f}%)")
            logger.warning(f"Largest class '{max_class_name}': {max_class_count:,} samples ({max_class_pct:.2f}%)")
            
            if self.balance_method is None:
                logger.warning("Consider setting balance_method to handle class imbalance")
                logger.warning("Suggested methods: 'class_weight' or 'smote'")
        else:
            logger.debug(f"Class balance is reasonable (imbalance ratio: {imbalance_ratio:.2f}) ✓")
        
        # Check compatibility with evaluation metric
        if self.eval_metric == 'auc' and not self.is_binary:
            logger.warning(f"AUC is not suitable as primary metric for multiclass problems")
            logger.warning("Consider using 'accuracy', 'balanced_accuracy', or 'f1' instead")
            validation_results['metric_warning'] = f"AUC not suitable for multiclass problems"
        
        # Feature analysis for classification
        logger.debug("Analyzing features for classification...")
        
        # Check for numeric features
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target]
        validation_results['numeric_feature_count'] = len(numeric_cols)
        
        # Check for categorical features
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col != self.target]
        validation_results['categorical_feature_count'] = len(cat_cols)
        
        logger.debug(f"Feature composition: {len(numeric_cols)} numeric, {len(cat_cols)} categorical")
        
        # Calculate class distribution entropy (diversity)
        try:
            from scipy.stats import entropy
            class_probs = self.class_distribution / self.class_distribution.sum()
            class_entropy = entropy(class_probs)
            max_entropy = entropy([1/len(class_probs)] * len(class_probs))  # Maximum possible entropy
            normalized_entropy = class_entropy / max_entropy
            
            validation_results['class_entropy'] = float(class_entropy)
            validation_results['normalized_entropy'] = float(normalized_entropy)
            
            logger.debug(f"Class distribution entropy: {class_entropy:.4f} (normalized: {normalized_entropy:.4f})")
            
            if normalized_entropy < 0.7 and len(class_probs) > 2:
                logger.warning(f"Class distribution is imbalanced (normalized entropy: {normalized_entropy:.4f})")
        except ImportError:
            logger.debug("Scipy not available, skipping entropy calculation")
        
        logger.info(f"Classification problem subtype: {self.problem_subtype}")
        logger.info("Class distribution:")
        for cls, count in self.class_distribution.items():
            percentage = count / len(self.df) * 100
            logger.info(f"  {cls}: {count:,} ({percentage:.1f}%)")
        
        validation_time = (datetime.now() - validation_start_time).total_seconds()
        logger.info(f"Data validation completed in {validation_time:.2f} seconds")
        
        
        return True
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Preprocess the data for classification with optional custom feature engineering.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Preprocessing data for classification...")
        preprocess_start_time = datetime.now()
        
        # Log preprocessing start
        logger.debug(f"Starting preprocessing for {self.problem_subtype} classification")
        logger.debug(f"Initial data shape: {self.df.shape}")
        
        # Call the parent method first to get basic preprocessing
        logger.debug("Calling parent class preprocessing method...")
        parent_start_time = datetime.now()
        X_train, X_test, y_train, y_test = super().preprocess_data()
        parent_time = (datetime.now() - parent_start_time).total_seconds()
        logger.debug(f"Parent preprocessing completed in {parent_time:.2f} seconds")
        
        # Log split sizes
        logger.debug(f"Train set: {X_train.shape[0]:,} rows ({X_train.shape[0]/self.df.shape[0]*100:.1f}% of data)")
        logger.debug(f"Test set: {X_test.shape[0]:,} rows ({X_test.shape[0]/self.df.shape[0]*100:.1f}% of data)")
        logger.debug(f"Features: {X_train.shape[1]:,} columns after preprocessing")
        
        # Encode target if it's categorical
        encode_start_time = datetime.now()
        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            logger.info("Encoding categorical target variable")
            logger.debug(f"Target dtype before encoding: {self.df[self.target].dtype}")
            
            # Log sample of original values
            sample_original = pd.Series(y_train).head(5).tolist()
            logger.debug(f"Original target sample (first 5): {sample_original}")
            
            # Encode target
            self.le = LabelEncoder()
            self.y_train = self.le.fit_transform(y_train)
            self.y_test = self.le.transform(y_test)
            self.class_names = list(self.le.classes_)
            
            # Log encoding mapping
            class_mapping = dict(zip(self.class_names, range(len(self.class_names))))
            logger.info(f"Target encoded with {len(class_mapping)} classes")
            logger.debug(f"Class mapping: {class_mapping}")
            
            # Log sample of encoded values
            sample_encoded = pd.Series(self.y_train).head(5).tolist()
            logger.debug(f"Encoded target sample (first 5): {sample_encoded}")
            
            # Store in metadata
            self.metadata['preprocessing']['class_encoding'] = class_mapping
            self.metadata['preprocessing']['class_names'] = self.class_names
        else:
            logger.info("Target is already numeric, no encoding needed")
            logger.debug(f"Target dtype: {pd.Series(y_train).dtype}")
            self.y_train = y_train
            self.y_test = y_test
            
            # Get unique classes
            unique_classes = sorted(pd.Series(y_train).unique())
            self.class_names = [str(cls) for cls in unique_classes]
            logger.debug(f"Class names (from numeric values): {self.class_names}")
            
            # Store in metadata
            self.metadata['preprocessing']['class_names'] = self.class_names
            
        encode_time = (datetime.now() - encode_start_time).total_seconds()
        logger.debug(f"Target encoding completed in {encode_time:.2f} seconds")
        
        # Check train/test distribution of classes
        logger.debug("Checking class distribution in train and test sets...")
        
        # Convert to numpy arrays for consistent handling
        y_train_array = self.y_train if isinstance(self.y_train, np.ndarray) else np.array(self.y_train)
        y_test_array = self.y_test if isinstance(self.y_test, np.ndarray) else np.array(self.y_test)
        
        # Count classes in train and test sets
        train_counts = np.bincount(y_train_array)
        test_counts = np.bincount(y_test_array)
        
        # Format percentages for logging
        train_percents = train_counts / len(y_train_array) * 100
        test_percents = test_counts / len(y_test_array) * 100
        
        # Log the distributions
        logger.debug("Class distribution in train set:")
        for i, (count, pct) in enumerate(zip(train_counts, train_percents)):
            class_name = self.class_names[i] if i < len(self.class_names) else str(i)
            logger.debug(f"  - Class '{class_name}': {count:,} ({pct:.1f}%)")
            
        logger.debug("Class distribution in test set:")
        for i, (count, pct) in enumerate(zip(test_counts, test_percents)):
            class_name = self.class_names[i] if i < len(self.class_names) else str(i)
            logger.debug(f"  - Class '{class_name}': {count:,} ({pct:.1f}%)")
            
        # Check for significant distribution differences
        distribution_diffs = np.abs(train_percents - test_percents)
        max_diff_idx = np.argmax(distribution_diffs)
        max_diff = distribution_diffs[max_diff_idx]
        
        if max_diff > 5:  # Arbitrary threshold of 5% difference
            logger.warning(f"Class distribution differs between train and test sets (max difference: {max_diff:.1f}%)")
            diff_class = self.class_names[max_diff_idx] if max_diff_idx < len(self.class_names) else str(max_diff_idx)
            logger.warning(f"Largest difference for class '{diff_class}': {train_percents[max_diff_idx]:.1f}% in train vs {test_percents[max_diff_idx]:.1f}% in test")
            
            # Store information about distribution difference
            self.metadata['preprocessing']['distribution_warning'] = {
                'max_difference_percent': float(max_diff),
                'class_with_max_diff': str(diff_class),
                'train_percent': float(train_percents[max_diff_idx]),
                'test_percent': float(test_percents[max_diff_idx])
            }
        else:
            logger.debug(f"Class distribution is consistent between train and test sets (max difference: {max_diff:.1f}%) ✓")
            
        # Add preprocessing metadata for classification
        self.metadata['preprocessing']['classification_specific'] = {
            'is_binary': self.is_binary,
            'problem_subtype': self.problem_subtype,
            'encoded_target': not pd.api.types.is_numeric_dtype(self.df[self.target]),
            'train_class_counts': train_counts.tolist(),
            'test_class_counts': test_counts.tolist()
        }
        
        preprocess_time = (datetime.now() - preprocess_start_time).total_seconds()
        logger.info(f"Data preprocessing for classification completed in {preprocess_time:.2f} seconds")
            
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
        With optional Random Search optimization for hyperparameters.
        
        Returns:
            dict: Trained models
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        logger.info("Training classification models...")
        train_start_time = datetime.now()
        
        if self.preprocessor is None:
            error_msg = "Data not preprocessed. Call preprocess_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log training dimensions
        logger.debug(f"Training data dimensions - X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        logger.debug(f"Problem type: {self.problem_subtype} classification with {len(self.class_names)} classes")
        
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
            training_metadata['data_sparsity'] = float(sparsity)
        
        # Define class weights if needed
        class_weights = None
        if self.balance_method == 'class_weight':
            logger.info("Calculating class weights for balanced training...")
            class_counts = np.bincount(self.y_train) if self.is_binary else np.unique(self.y_train, return_counts=True)[1]
            total = len(self.y_train)
            class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
            logger.info(f"Using class weights:")
            for i, weight in class_weights.items():
                class_name = self.class_names[i] if i < len(self.class_names) else str(i)
                logger.info(f"  - Class '{class_name}': weight={weight:.4f}")
            
            training_metadata['class_weights'] = class_weights
        
        # Apply SMOTE if requested
        if self.balance_method == 'smote':
            logger.info("Applying SMOTE to balance classes...")
            try:
                from imblearn.over_sampling import SMOTE
                
                # Log memory before SMOTE
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / (1024 * 1024)
                    logger.debug(f"Memory before SMOTE: {memory_before:.2f} MB")
                except ImportError:
                    pass
                
                smote_start = datetime.now()
                smote = SMOTE(random_state=self.random_state)
                X_train_processed, y_train_resampled = smote.fit_resample(X_train_processed, self.y_train)
                smote_time = (datetime.now() - smote_start).total_seconds()
                
                # Log SMOTE results
                logger.info(f"SMOTE resampling completed in {smote_time:.2f} seconds")
                logger.info(f"Data shape after SMOTE: {X_train_processed.shape} ({X_train_processed.shape[0] - len(self.y_train):+,} samples)")
                
                # Log memory after SMOTE
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_after = process.memory_info().rss / (1024 * 1024)
                    memory_diff = memory_after - memory_before
                    logger.debug(f"Memory after SMOTE: {memory_after:.2f} MB (change: {memory_diff:+.2f} MB)")
                except ImportError:
                    pass
                
                # Log new class distribution
                unique, counts = np.unique(y_train_resampled, return_counts=True)
                class_dist_after_smote = {}
                logger.info("Class distribution after SMOTE:")
                for cls, count in zip(unique, counts):
                    cls_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
                    class_dist_after_smote[str(cls_name)] = int(count)
                    logger.info(f"  - Class '{cls_name}': {count:,} ({count/len(y_train_resampled)*100:.1f}%)")
                
                training_metadata['smote'] = {
                    'applied': True,
                    'time_seconds': smote_time,
                    'samples_before': int(len(self.y_train)),
                    'samples_after': int(len(y_train_resampled)),
                    'increase_percent': float((len(y_train_resampled) - len(self.y_train)) / len(self.y_train) * 100),
                    'class_distribution_after': class_dist_after_smote
                }
            except Exception as e:
                error_msg = f"Error applying SMOTE: {str(e)}. Falling back to original data."
                logger.error(error_msg, exc_info=True)
                logger.debug(f"SMOTE traceback: {traceback.format_exc()}")
                y_train_resampled = self.y_train
                training_metadata['smote'] = {
                    'applied': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        else:
            logger.debug("No SMOTE resampling requested")
            y_train_resampled = self.y_train
            training_metadata['smote'] = {'applied': False}
        
        # Get model configurations from config
        model_params = self.model_config.get('models', {}).get('parameters', {})
        enabled_models = self.model_config.get('models', {}).get('enabled', [])
        
        logger.debug(f"Model parameters from config: {model_params}")
        logger.debug(f"Enabled models: {enabled_models}")
        
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
        
        # Log configuration for each model
        for name, classifier in all_classifiers.items():
            logger.debug(f"Model '{name}' configuration: {classifier.get_params()}")
        
        # Add more complex models for smaller datasets
        dataset_size = len(self.X_train)
        if dataset_size < 10000:
            logger.info(f"Dataset size ({dataset_size:,} rows) < 10,000, adding SVC and MLP models")
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
            logger.debug(f"SVC configuration: {all_classifiers['svc'].get_params()}")
            logger.debug(f"MLP configuration: {all_classifiers['mlp'].get_params()}")
        else:
            logger.info(f"Dataset size ({dataset_size:,} rows) >= 10,000, skipping SVC and MLP models")
        
        classifiers = {k: v for k, v in all_classifiers.items() if k in enabled_models}
        logger.info(f"Training {len(classifiers)} models: {', '.join(classifiers.keys())}")

        # Train all models
        self.models = {}
        training_metadata['model_training_times'] = {}
        
        # First, train models with regular parameters
        for name, classifier in classifiers.items():
            try:
                logger.info(f"Training {name}...")
                model_metadata = {
                    'model_type': str(type(classifier)),
                    'parameters': str(classifier.get_params())
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
                
                try:
                    model = classifier.fit(X_train_processed, y_train_resampled)
                    
                    train_time = (datetime.now() - train_start).total_seconds()
                    logger.info(f"Training {name} completed in {train_time:.2f} seconds")
                    
                    # Log model details
                    if hasattr(model, 'feature_importances_'):
                        logger.debug(f"Model {name} has feature importance information")
                        top_features = sorted(zip(range(X_train_processed.shape[1]), model.feature_importances_), 
                                              key=lambda x: x[1], reverse=True)[:10]
                        
                        logger.debug(f"Top 10 feature importances for {name}:")
                        for idx, importance in top_features:
                            logger.debug(f"  - Feature_{idx}: {importance:.4f}")
                            
                        model_metadata['feature_importances'] = {f"Feature_{idx}": float(imp) for idx, imp in top_features}
                    
                    # Check for convergence warnings
                    if hasattr(model, 'n_iter_') and name == 'logistic_regression':
                        logger.debug(f"Logistic regression iterations: {model.n_iter_}")
                        if model.n_iter_ >= 1000:  # max_iter value
                            logger.warning(f"Logistic regression may not have converged (iterations: {model.n_iter_})")
                            model_metadata['convergence_warning'] = True
                    
                    # Log memory after training
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_after = process.memory_info().rss / (1024 * 1024)
                        memory_diff = memory_after - memory_before
                        logger.debug(f"Memory after training {name}: {memory_after:.2f} MB (change: {memory_diff:+.2f} MB)")
                    except ImportError:
                        pass
                    
                    # Store the model
                    self.models[name] = model
                    
                    # Update metadata
                    model_metadata['training_time_seconds'] = train_time
                    model_metadata['trained_successfully'] = True
                    model_metadata['hyperparameter_optimization'] = False
                    
                    # Track model complexity
                    complexity_metrics = self._get_model_complexity_metrics(model)
                    if complexity_metrics:
                        model_metadata['complexity_metrics'] = complexity_metrics
                        logger.debug(f"Model complexity metrics: {complexity_metrics}")
                    
                    training_metadata['models'][name] = model_metadata
                    training_metadata['model_training_times'][name] = train_time
                
                except Exception as model_error:
                    train_time = (datetime.now() - train_start).total_seconds()
                    logger.error(f"Error during {name} model training: {str(model_error)}")
                    logger.debug(f"Model training error traceback: {traceback.format_exc()}")
                    training_metadata['models'][name] = {
                        'trained_successfully': False,
                        'error': str(model_error),
                        'training_attempted_seconds': train_time,
                        'traceback': traceback.format_exc()
                    }
                
            except Exception as e:
                error_msg = f"Failed to initialize {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                training_metadata['models'][name] = {
                    'trained_successfully': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Next, apply Random Search optimization if enabled
        random_search_config = self.model_config.get('random_search', {})
        use_random_search = random_search_config.get('enabled', False)
        
        if use_random_search:
            logger.info("Random Search optimization enabled for classification models")
            n_iter = random_search_config.get('n_iter', 20)
            cv = random_search_config.get('cv', 5)
            verbose = random_search_config.get('verbose', 1)
            rs_models_config = random_search_config.get('models', {})
            
            logger.debug(f"Random Search parameters: n_iter={n_iter}, cv={cv}, verbose={verbose}")
            logger.debug(f"Models to optimize: {list(rs_models_config.keys())}")
            
            # Apply Random Search to configured models
            for model_name, model_config in rs_models_config.items():
                if model_config.get('enabled', False) and model_name in classifiers:
                    logger.info(f"Performing Random Search optimization for {model_name}")
                    
                    # Get base model
                    base_model = classifiers[model_name]
                    
                    # Get parameter distributions
                    param_distributions = model_config.get('parameters', {})
                    
                    # Verify param_distributions is suitable for RandomizedSearchCV
                    if not param_distributions:
                        logger.warning(f"No parameter distributions defined for {model_name}, skipping Random Search")
                        continue
                    
                    logger.debug(f"Parameter distributions for {model_name}: {param_distributions}")
                    
                    try:
                        # Select appropriate scoring metric based on class balance
                        scorer = 'accuracy'
                        if hasattr(self, 'class_distribution'):
                            min_class_pct = (self.class_distribution.min() / len(self.df)) * 100
                            if min_class_pct < 20:  # If imbalanced
                                logger.info("Using 'balanced_accuracy' for optimization due to class imbalance")
                                scorer = 'balanced_accuracy'
                        
                        logger.debug(f"Using '{scorer}' as optimization metric")
                        
                        # Log memory before optimization
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_before = process.memory_info().rss / (1024 * 1024)
                            logger.debug(f"Memory before Random Search for {model_name}: {memory_before:.2f} MB")
                        except ImportError:
                            pass
                        
                        # Create RandomizedSearchCV
                        logger.debug(f"Creating RandomizedSearchCV for {model_name}")
                        random_search = RandomizedSearchCV(
                            base_model,
                            param_distributions=param_distributions,
                            n_iter=n_iter,
                            cv=cv,
                            scoring=scorer,
                            random_state=self.random_state,
                            verbose=verbose,
                            n_jobs=-1
                        )
                        
                        # Time the optimization
                        train_start = datetime.now()
                        
                        # Fit on processed data
                        logger.info(f"Starting Random Search with {n_iter} iterations and {cv}-fold CV")
                        random_search.fit(X_train_processed, y_train_resampled)
                        
                        # Add optimized model with a distinct name
                        optimized_name = f"{model_name}_optimized"
                        self.models[optimized_name] = random_search.best_estimator_
                        
                        # Calculate training time
                        train_time = (datetime.now() - train_start).total_seconds()
                        logger.info(f"Random Search for {model_name} completed in {train_time:.2f} seconds")
                        logger.info(f"Best parameters: {random_search.best_params_}")
                        
                        # Log the CV results
                        cv_results = pd.DataFrame(random_search.cv_results_)
                        best_score = random_search.best_score_
                        logger.info(f"Best CV score ({scorer}): {best_score:.4f}")
                        
                        # Log CV result statistics
                        mean_scores = cv_results['mean_test_score']
                        logger.debug(f"CV scores - Mean: {mean_scores.mean():.4f}, Std: {mean_scores.std():.4f}")
                        logger.debug(f"CV scores - Min: {mean_scores.min():.4f}, Max: {mean_scores.max():.4f}")
                        
                        # Log memory after optimization
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_after = process.memory_info().rss / (1024 * 1024)
                            memory_diff = memory_after - memory_before
                            logger.debug(f"Memory after Random Search for {model_name}: {memory_after:.2f} MB (change: {memory_diff:+.2f} MB)")
                        except ImportError:
                            pass
                        
                        # Log feature importances if available
                        if hasattr(random_search.best_estimator_, 'feature_importances_'):
                            top_features = sorted(zip(range(X_train_processed.shape[1]), 
                                                  random_search.best_estimator_.feature_importances_), 
                                               key=lambda x: x[1], reverse=True)[:10]
                            
                            logger.debug(f"Top 10 feature importances for optimized {model_name}:")
                            for idx, importance in top_features:
                                logger.debug(f"  - Feature_{idx}: {importance:.4f}")
                        
                        # Update metadata
                        optimized_metadata = {
                            'model_type': str(type(random_search.best_estimator_)),
                            'parameters': str(random_search.best_params_),
                            'training_time_seconds': train_time,
                            'trained_successfully': True,
                            'hyperparameter_optimization': True,
                            'optimization_method': 'random_search',
                            'optimization_metric': scorer,
                            'n_iter': n_iter,
                            'cv_folds': cv,
                            'best_score': float(best_score),
                            'cv_results_summary': {
                                'mean_score': float(mean_scores.mean()),
                                'std_score': float(mean_scores.std()),
                                'min_score': float(mean_scores.min()),
                                'max_score': float(mean_scores.max())
                            }
                        }
                        
                        # Add feature importances if available
                        if hasattr(random_search.best_estimator_, 'feature_importances_'):
                            optimized_metadata['feature_importances'] = {
                                f"Feature_{idx}": float(imp) for idx, imp in top_features
                            }
                        
                        # Track model complexity
                        complexity_metrics = self._get_model_complexity_metrics(random_search.best_estimator_)
                        if complexity_metrics:
                            optimized_metadata['complexity_metrics'] = complexity_metrics
                        
                        training_metadata['models'][optimized_name] = optimized_metadata
                        training_metadata['model_training_times'][optimized_name] = train_time
                        
                    except Exception as e:
                        error_msg = f"Failed to optimize {model_name} with Random Search: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        logger.debug(f"Random Search error traceback: {traceback.format_exc()}")
                        training_metadata['models'][f"{model_name}_optimized"] = {
                            'trained_successfully': False,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }
        
        # Count successful models
        total_models = len(self.models)
        successful_models = sum(1 for name, metadata in training_metadata['models'].items() 
                                if metadata.get('trained_successfully', False))
        
        # Log models sorted by training time
        training_times = sorted(training_metadata['model_training_times'].items(), key=lambda x: x[1])
        logger.debug("Models by training time (fastest to slowest):")
        for model_name, time_sec in training_times:
            logger.debug(f"  - {model_name}: {time_sec:.2f} seconds")
        
        # Log performance comparison between base and optimized models if applicable
        base_and_optimized = [name for name in self.models if '_optimized' in name]
        if base_and_optimized:
            logger.debug("Optimized models created:")
            for opt_name in base_and_optimized:
                base_name = opt_name.replace('_optimized', '')
                logger.debug(f"  - {opt_name} (based on {base_name})")
        
        total_training_time = (datetime.now() - train_start_time).total_seconds()
        logger.info(f"Successfully trained {successful_models}/{total_models} models in {total_training_time:.2f} seconds")
        
        # Update metadata
        training_metadata['total_training_time_seconds'] = total_training_time
        training_metadata['models_attempted'] = total_models
        training_metadata['models_successful'] = successful_models
        self.metadata['models'] = training_metadata
        
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
            if hasattr(model.coef_, 'shape'):
                metrics['n_coefficients'] = model.coef_.size
                metrics['n_nonzero_coefficients'] = np.count_nonzero(model.coef_)
            
        return metrics

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
        
        logger.info("Evaluating classification models...")
        eval_start_time = datetime.now()
        
        # Log evaluation data
        logger.debug(f"Evaluating {len(self.models)} models on test data")
        logger.debug(f"Test data size: {self.X_test.shape[0]:,} rows, {self.X_test.shape[1]:,} columns")
        logger.debug(f"Problem type: {self.problem_subtype} classification with {len(self.class_names)} classes")
        
        # Get primary metric function
        primary_metric = self._get_primary_metric()
        primary_metric_name = self.eval_metric
        
        logger.debug(f"Primary evaluation metric: {primary_metric_name}")
        
        evaluation_metadata = {'models': {}}
        
        # Transform test data
        preproc_start = datetime.now()
        X_test_processed = self.preprocessor.transform(self.X_test)
        preproc_time = (datetime.now() - preproc_start).total_seconds()
        logger.debug(f"Test data preprocessing completed in {preproc_time:.2f} seconds")
        
        results = []
        
        # Create a table to store detailed metrics for each model
        detailed_metrics = []
        
        for name, model in self.models.items():
            try:
                logger.info(f"Evaluating {name}...")
                model_eval_metadata = {}
                eval_start = datetime.now()
                
                # Make predictions
                train_pred_start = datetime.now()
                y_train_pred = model.predict(self.preprocessor.transform(self.X_train))
                train_pred_time = (datetime.now() - train_pred_start).total_seconds()
                logger.debug(f"Train prediction time: {train_pred_time:.4f} seconds")
                
                test_pred_start = datetime.now()
                y_test_pred = model.predict(X_test_processed)
                test_pred_time = (datetime.now() - test_pred_start).total_seconds()
                logger.debug(f"Test prediction time: {test_pred_time:.4f} seconds")
                
                # Calculate metrics appropriate for all classification problems
                model_eval_metadata['prediction_time'] = {
                    'train_seconds': train_pred_time,
                    'test_seconds': test_pred_time,
                    'test_samples_per_second': len(self.y_test) / test_pred_time
                }
                
                # Log prediction timing
                ms_per_sample = test_pred_time * 1000 / len(self.y_test)
                logger.debug(f"Prediction speed: {ms_per_sample:.2f} ms per sample")
                
                # Calculate accuracy
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                
                # Calculate balanced accuracy
                train_bal_accuracy = balanced_accuracy_score(self.y_train, y_train_pred)
                test_bal_accuracy = balanced_accuracy_score(self.y_test, y_test_pred)
                
                # Calculate primary metric
                train_primary = primary_metric(self.y_train, y_train_pred)
                test_primary = primary_metric(self.y_test, y_test_pred)
                
                # Store results
                result = {
                    'model': name,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_balanced_accuracy': train_bal_accuracy,
                    'test_balanced_accuracy': test_bal_accuracy,
                    f'train_{primary_metric_name}': train_primary,
                    f'test_{primary_metric_name}': test_primary
                }
                
                # Calculate prediction time per sample
                result['prediction_ms_per_sample'] = ms_per_sample
                
                # Check for overfitting
                accuracy_drop = train_accuracy - test_accuracy
                overfitting_detected = accuracy_drop > 0.1  # Arbitrary threshold
                
                if overfitting_detected:
                    logger.warning(f"{name}: Possible overfitting - Accuracy drop: {accuracy_drop:.4f} (train: {train_accuracy:.4f}, test: {test_accuracy:.4f})")
                
                # Additional metrics for binary classification
                if self.is_binary:
                    # Precision, recall, F1
                    test_precision = precision_score(self.y_test, y_test_pred)
                    test_recall = recall_score(self.y_test, y_test_pred)
                    test_f1 = f1_score(self.y_test, y_test_pred)
                    
                    result['test_precision'] = test_precision
                    result['test_recall'] = test_recall
                    result['test_f1'] = test_f1
                    
                    # Matthews Correlation Coefficient (MCC)
                    test_mcc = matthews_corrcoef(self.y_test, y_test_pred)
                    result['test_mcc'] = test_mcc
                    
                    # Log loss if probabilities are available
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_test_proba = model.predict_proba(X_test_processed)
                            test_log_loss = log_loss(self.y_test, y_test_proba)
                            result['test_log_loss'] = test_log_loss
                        except Exception as e:
                            logger.debug(f"Could not calculate log loss: {str(e)}")
                    
                    # AUC for binary problems if classifier supports predict_proba
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_test_prob = model.predict_proba(X_test_processed)[:, 1]
                            auc = roc_auc_score(self.y_test, y_test_prob)
                            result['test_auc'] = auc
                            
                            # Average precision score (area under precision-recall curve)
                            try:
                                avg_precision = average_precision_score(self.y_test, y_test_prob)
                                result['test_avg_precision'] = avg_precision
                            except Exception as e:
                                logger.debug(f"Could not calculate average precision: {str(e)}")
                            
                        except Exception as e:
                            logger.warning(f"Could not calculate AUC for {name}: {str(e)}")
                            result['test_auc'] = np.nan
                
                # Additional metrics for multiclass problems
                else:
                    # Precision, recall, F1 (macro averaged)
                    test_precision_macro = precision_score(self.y_test, y_test_pred, average='macro')
                    test_recall_macro = recall_score(self.y_test, y_test_pred, average='macro')
                    test_f1_macro = f1_score(self.y_test, y_test_pred, average='macro')
                    
                    result['test_precision_macro'] = test_precision_macro
                    result['test_recall_macro'] = test_recall_macro
                    result['test_f1_macro'] = test_f1_macro
                    
                    # Precision, recall, F1 (weighted averaged)
                    test_precision_weighted = precision_score(self.y_test, y_test_pred, average='weighted')
                    test_recall_weighted = recall_score(self.y_test, y_test_pred, average='weighted')
                    test_f1_weighted = f1_score(self.y_test, y_test_pred, average='weighted')
                    
                    result['test_precision_weighted'] = test_precision_weighted
                    result['test_recall_weighted'] = test_recall_weighted
                    result['test_f1_weighted'] = test_f1_weighted
                    
                    # Log loss if probabilities are available
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_test_proba = model.predict_proba(X_test_processed)
                            test_log_loss = log_loss(self.y_test, y_test_proba)
                            result['test_log_loss'] = test_log_loss
                        except Exception as e:
                            logger.debug(f"Could not calculate log loss: {str(e)}")
                
                # Add result to results list
                results.append(result)
                
                # Calculate confusion matrix
                cm = confusion_matrix(self.y_test, y_test_pred)
                model_eval_metadata['confusion_matrix'] = cm.tolist()
                
                # Format confusion matrix for logging
                cm_str = np.array2string(cm, separator=', ')
                logger.debug(f"Confusion matrix:\n{cm_str}")
                
                # Check for class-specific issues
                class_recall = np.diag(cm) / np.sum(cm, axis=1)
                for i, recall in enumerate(class_recall):
                    if recall < 0.5:  # Arbitrary threshold
                        class_name = self.class_names[i] if i < len(self.class_names) else str(i)
                        logger.warning(f"{name}: Low recall ({recall:.4f}) for class '{class_name}'")
                
                # Calculate detailed classification report
                try:
                    target_names = [str(name) for name in self.class_names]
                    cls_report = classification_report(self.y_test, y_test_pred, 
                                                      target_names=target_names, 
                                                      output_dict=True)
                    model_eval_metadata['classification_report'] = cls_report
                    
                    # Log per-class metrics
                    logger.debug(f"Per-class metrics for {name}:")
                    for cls, metrics in cls_report.items():
                        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                            precision = metrics['precision']
                            recall = metrics['recall']
                            f1 = metrics['f1-score']
                            support = metrics['support']
                            logger.debug(f"  - Class '{cls}': Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
                    
                except Exception as e:
                    logger.warning(f"Could not generate classification report for {name}: {str(e)}")
                
                # Update metadata with metrics
                model_eval_metadata['metrics'] = {}
                
                # Store all metrics from result dict
                for key, value in result.items():
                    if key != 'model':
                        model_eval_metadata['metrics'][key] = float(value)
                
                # Store detailed metrics for each model
                detailed_metrics.append({
                    'model': name,
                    'accuracy': test_accuracy,
                    'balanced_accuracy': test_bal_accuracy,
                    'primary_metric': test_primary,
                    'primary_metric_name': primary_metric_name,
                    'overfitting_detected': overfitting_detected,
                    'accuracy_drop': accuracy_drop
                })
                
                # Store overfitting analysis
                model_eval_metadata['overfitting_analysis'] = {
                    'accuracy_drop': float(accuracy_drop),
                    'overfitting_detected': overfitting_detected
                }
                
                evaluation_metadata['models'][name] = model_eval_metadata
                
                # Calculate evaluation time
                eval_time = (datetime.now() - eval_start).total_seconds()
                model_eval_metadata['evaluation_time_seconds'] = eval_time
                
                # Log results
                logger.info(f"  {name}:")
                logger.info(f"    Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
                
                if self.is_binary:
                    precision = precision_score(self.y_test, y_test_pred)
                    recall = recall_score(self.y_test, y_test_pred)
                    f1 = f1_score(self.y_test, y_test_pred)
                    logger.info(f"    Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}")
                    
                    if 'test_auc' in result and not pd.isna(result['test_auc']):
                        logger.info(f"    Test AUC: {result['test_auc']:.4f}")
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
                logger.debug(f"Evaluation error traceback: {traceback.format_exc()}")
                evaluation_metadata['models'][name] = {
                    'evaluation_error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Convert to DataFrame
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        evaluation_metadata['results_summary'] = self.results.to_dict()
        
        # Add detailed metrics summary
        evaluation_metadata['detailed_metrics'] = detailed_metrics
        
        # Log metrics ranges across all models
        if not self.results.empty:
            logger.debug("Metrics ranges across all models:")
            
            for col in self.results.columns:
                if col != 'model' and self.results[col].dtype in [np.float64, np.int64]:
                    min_val = self.results[col].min()
                    max_val = self.results[col].max()
                    mean_val = self.results[col].mean()
                    logger.debug(f"  - {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, range={max_val-min_val:.4f}")
        
        # Identify best model based on primary metric
        if not self.results.empty:
            # Adjust metric name for multiclass case
            if not self.is_binary and primary_metric_name in ['precision', 'recall', 'f1']:
                metric_col = f'test_{primary_metric_name}_macro'
                logger.debug(f"Using '{metric_col}' as primary metric for multiclass")
            else:
                metric_col = f'test_{primary_metric_name}'
                logger.debug(f"Using '{metric_col}' as primary metric")
            
            # Find best model
            if metric_col in self.results.columns:
                best_idx = self.results[metric_col].idxmax()
                best_model_name = self.results.loc[best_idx, 'model']
                best_metric_value = self.results.loc[best_idx, metric_col]
                
                self.best_model = self.models[best_model_name]
                logger.info(f"\nBest model: {best_model_name} (Test {primary_metric_name} = {best_metric_value:.4f})")
                
                # Log all metrics for best model
                logger.debug(f"All metrics for best model ({best_model_name}):")
                for col in self.results.columns:
                    if col != 'model':
                        val = self.results.loc[best_idx, col]
                        if isinstance(val, (int, float)):
                            logger.debug(f"  - {col}: {val:.4f}")
                
                # Check how much better than baseline (if logistic regression is available)
                if 'logistic_regression' in self.models:
                    try:
                        baseline_idx = self.results[self.results['model'] == 'logistic_regression'].index[0]
                        baseline_value = self.results.loc[baseline_idx, metric_col]
                        improvement_pct = (best_metric_value - baseline_value) / baseline_value * 100
                        
                        logger.info(f"Improvement over logistic regression baseline: {improvement_pct:.2f}% ({best_metric_value:.4f} vs {baseline_value:.4f})")
                        
                        evaluation_metadata['improvement_over_baseline'] = {
                            'baseline_model': 'logistic_regression',
                            'baseline_value': float(baseline_value),
                            'best_value': float(best_metric_value), 
                            'improvement_percent': float(improvement_pct)
                        }
                    except Exception as e:
                        logger.debug(f"Could not calculate improvement over baseline: {str(e)}")
                
                # Get predictions from the best model
                logger.debug("Generating predictions with best model")
                pred_start = datetime.now()
                best_test_predictions = self.best_model.predict(X_test_processed)
                pred_time = (datetime.now() - pred_start).total_seconds()
                logger.debug(f"Generated predictions in {pred_time:.4f} seconds")
                
                # Store best model predictions
                self._store_best_model_predictions(best_model_name, best_test_predictions)

                # Update metadata
                self.metadata['best_model']['name'] = best_model_name
                self.metadata['best_model'][metric_col] = float(best_metric_value)
                
                # Store all metrics for best model
                best_metrics = {}
                for col in self.results.columns:
                    if col != 'model':
                        val = self.results.loc[best_idx, col]
                        if isinstance(val, (int, float)):
                            best_metrics[col] = float(val)
                
                self.metadata['best_model']['metrics'] = best_metrics
                
                # Add more metrics based on problem type
                if self.is_binary and 'test_auc' in self.results.columns:
                    auc_value = self.results.loc[best_idx, 'test_auc']
                    if not pd.isna(auc_value):
                        self.metadata['best_model']['metrics']['test_auc'] = float(auc_value)
            else:
                logger.warning(f"Primary metric '{metric_col}' not found in results. Using accuracy instead.")
                best_idx = self.results['test_accuracy'].idxmax()
                best_model_name = self.results.loc[best_idx, 'model']
                best_accuracy = self.results.loc[best_idx, 'test_accuracy']
                
                self.best_model = self.models[best_model_name]
                logger.info(f"\nBest model: {best_model_name} (Test accuracy = {best_accuracy:.4f})")
                
                # Get predictions from the best model
                best_test_predictions = self.best_model.predict(X_test_processed)
                self._store_best_model_predictions(best_model_name, best_test_predictions)
                
                # Update metadata
                self.metadata['best_model']['name'] = best_model_name
                self.metadata['best_model']['test_accuracy'] = float(best_accuracy)
        
        # Calculate overall evaluation time
        eval_total_time = (datetime.now() - eval_start_time).total_seconds()
        logger.info(f"Model evaluation completed in {eval_total_time:.2f} seconds")
        
        # Store total evaluation time in metadata
        evaluation_metadata['total_evaluation_time_seconds'] = eval_total_time
        
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
        store_start = datetime.now()
        
        # Get the original class names
        if hasattr(self, 'le') and self.le is not None:
            # Get encoded predictions and actual values
            y_test_encoded = self.y_test
            predictions_encoded = predictions
            
            logger.debug(f"Decoding predictions from encoded values using LabelEncoder")
            
            # Decode them back to original labels
            try:
                y_test_decoded = self.le.inverse_transform(y_test_encoded)
                predictions_decoded = self.le.inverse_transform(predictions_encoded)
                
                logger.debug(f"Successfully decoded {len(y_test_decoded)} actual values and {len(predictions_decoded)} predictions")
                
                # Create a DataFrame with actual and predicted values (using decoded values)
                prediction_df = pd.DataFrame({
                    'actual': y_test_decoded,
                    'predicted': predictions_decoded
                })
                
                # Log to confirm correct decoding
                logger.debug(f"Decoded classes - first 5 samples:")
                for i in range(min(5, len(y_test_decoded))):
                    logger.debug(f"  Actual: {y_test_decoded[i]}, Predicted: {predictions_decoded[i]}")
                    
            except Exception as e:
                logger.error(f"Failed to decode class labels: {str(e)}")
                logger.debug(f"Decoding error traceback: {traceback.format_exc()}")
                
                # Fallback to encoded values
                logger.debug("Falling back to encoded values")
                prediction_df = pd.DataFrame({
                    'actual': y_test_encoded,
                    'predicted': predictions_encoded
                })
        else:
            # No encoding was done, use original values
            logger.debug("No label encoding was applied, using original values")
            prediction_df = pd.DataFrame({
                'actual': self.y_test,
                'predicted': predictions
            })
        
        # Calculate the total number of predictions
        total_predictions = len(prediction_df)
        logger.debug(f"Total predictions: {total_predictions:,}")
        
        # Add correct/incorrect column
        prediction_df['correct'] = prediction_df['actual'] == prediction_df['predicted']
        correct_count = prediction_df['correct'].sum()
        accuracy = correct_count / total_predictions
        
        logger.debug(f"Prediction accuracy: {accuracy:.4f} ({correct_count:,} correct out of {total_predictions:,})")
        
        # Define incorrect_predictions AFTER creating the 'correct' column
        incorrect_predictions = prediction_df[~prediction_df['correct']]
        incorrect_count = len(incorrect_predictions)
        
        logger.debug(f"Incorrect predictions: {incorrect_count:,} ({incorrect_count/total_predictions*100:.2f}%)")
        
        # If there's an index in the original test data, try to preserve it
        if hasattr(self.y_test, 'index') and self.y_test.index is not None:
            logger.debug("Preserving original test data index")
            prediction_df.index = self.y_test.index
        
        # Get class labels if available
        class_labels = None
        try:
            if hasattr(self.preprocessor, 'named_transformers_') and hasattr(self.preprocessor.named_transformers_, 'cat'):
                if hasattr(self.preprocessor.named_transformers_.cat, 'named_steps'):
                    if hasattr(self.preprocessor.named_transformers_.cat.named_steps, 'encoder'):
                        encoder = self.preprocessor.named_transformers_.cat.named_steps.encoder
                        if hasattr(encoder, 'categories_'):
                            class_labels = encoder.categories_
                            logger.debug(f"Found class labels from preprocessor: {class_labels}")
        except Exception as e:
            logger.debug(f"Could not extract class labels from preprocessor: {str(e)}")
        
        # Save confusion matrix
        from sklearn.metrics import confusion_matrix
        
        logger.debug("Calculating confusion matrix")
        cm = confusion_matrix(self.y_test, predictions)
        
        # Log confusion matrix
        if cm.shape[0] <= 10:  # Only log full matrix if not too large
            logger.debug(f"Confusion matrix:\n{cm}")
        else:
            logger.debug(f"Confusion matrix shape: {cm.shape} (too large to display)")
        
        # Analyze confusion matrix for patterns
        try:
            if len(self.class_names) <= 10:  # Only analyze if not too many classes
                class_precisions = np.diag(cm) / np.sum(cm, axis=0)
                class_recalls = np.diag(cm) / np.sum(cm, axis=1)
                
                logger.debug("Per-class metrics from confusion matrix:")
                for i, (precision, recall) in enumerate(zip(class_precisions, class_recalls)):
                    if i < len(self.class_names):
                        class_name = self.class_names[i]
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        logger.debug(f"  - Class '{class_name}': Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                        
                        if precision < 0.7 and recall < 0.7:  # Arbitrary threshold
                            logger.warning(f"Poor performance for class '{class_name}': Precision={precision:.4f}, Recall={recall:.4f}")
        except Exception as e:
            logger.debug(f"Could not analyze confusion matrix in detail: {str(e)}")
        
        # Convert confusion matrix to a list for JSON serialization
        cm_list = cm.tolist()
        
        # Calculate counts and percentages of correct/incorrect predictions
        prediction_stats = {
            'correct_count': int(correct_count),
            'incorrect_count': int(total_predictions - correct_count),
            'total_count': int(total_predictions),
            'accuracy': float(accuracy),
            'confusion_matrix': cm_list
        }
        
        # Add class distribution if available
        actual_class_counts = pd.Series(self.y_test).value_counts().to_dict()
        predicted_class_counts = pd.Series(predictions).value_counts().to_dict()
        
        # Convert to string keys for JSON serialization
        prediction_stats['actual_class_distribution'] = {str(k): int(v) for k, v in actual_class_counts.items()}
        prediction_stats['predicted_class_distribution'] = {str(k): int(v) for k, v in predicted_class_counts.items()}
        
        # Log class distributions
        logger.debug("Actual class distribution:")
        for cls, count in sorted(actual_class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_predictions * 100
            logger.debug(f"  - Class {cls}: {count:,} ({pct:.2f}%)")
            
        logger.debug("Predicted class distribution:")
        for cls, count in sorted(predicted_class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_predictions * 100
            logger.debug(f"  - Class {cls}: {count:,} ({pct:.2f}%)")
        
        # Store class labels if available
        if class_labels is not None:
            prediction_stats['class_labels'] = class_labels
        
        # Store prediction statistics in metadata
        self.metadata['best_model']['prediction_stats'] = prediction_stats
        
        # Analyze misclassifications by class
        if incorrect_count > 0:
            logger.debug("Analyzing misclassifications by class...")
            misclass_by_actual = incorrect_predictions.groupby('actual').size().to_dict()
            misclass_by_predicted = incorrect_predictions.groupby('predicted').size().to_dict()
            
            # Convert to proper format for JSON
            misclass_by_actual = {str(k): int(v) for k, v in misclass_by_actual.items()}
            misclass_by_predicted = {str(k): int(v) for k, v in misclass_by_predicted.items()}
            
            # Store in metadata
            prediction_stats['misclassifications_by_actual'] = misclass_by_actual
            prediction_stats['misclassifications_by_predicted'] = misclass_by_predicted
            
            # Find most common misclassification pairs
            pair_counts = incorrect_predictions.groupby(['actual', 'predicted']).size()
            if len(pair_counts) > 0:
                top_pairs = pair_counts.sort_values(ascending=False).head(5)
                logger.debug("Most common misclassifications (actual → predicted):")
                for (actual, predicted), count in top_pairs.items():
                    logger.debug(f"  - {actual} → {predicted}: {count:,} times")
                
                # Add to metadata
                prediction_stats['top_misclassification_pairs'] = [
                    {'actual': str(actual), 'predicted': str(predicted), 'count': int(count)}
                    for (actual, predicted), count in top_pairs.items()
                ]
        
        # Store a sample of predictions in metadata
        logger.debug("Sampling predictions for metadata storage...")
        if len(prediction_df) > max_predictions:
            # For classification, stratify by correct/incorrect and class
            # We already defined incorrect_predictions above, so no need to redefine
            correct_predictions = prediction_df[prediction_df['correct']]
            
            logger.debug(f"Taking a stratified sample from {len(correct_predictions):,} correct and {len(incorrect_predictions):,} incorrect predictions")
            
            # Take more incorrect examples as they're more interesting
            incorrect_sample_size = min(max_predictions // 2, len(incorrect_predictions))
            correct_sample_size = max_predictions - incorrect_sample_size
            
            logger.debug(f"Sample sizes - incorrect: {incorrect_sample_size:,}, correct: {correct_sample_size:,}")
            
            # Sample from incorrect predictions, stratifying by actual class
            # This ensures we get examples of different error types
            if len(incorrect_predictions) > 0:
                # Try to stratify by actual class to get diverse error examples
                try:
                    if 'actual' in incorrect_predictions.columns:
                        sampled_incorrect = []
                        incorrect_by_class = incorrect_predictions.groupby('actual')
                        
                        # Calculate samples per class proportionally
                        total_classes = len(incorrect_by_class)
                        if total_classes > 0:
                            base_samples_per_class = max(1, incorrect_sample_size // total_classes)
                            logger.debug(f"Sampling approximately {base_samples_per_class} incorrect predictions per class")
                            
                            remaining_samples = incorrect_sample_size
                            for cls, group in incorrect_by_class:
                                # Take proportional samples from each class
                                class_sample_size = min(len(group), base_samples_per_class)
                                if class_sample_size > 0:
                                    class_sample = group.sample(n=class_sample_size, random_state=42)
                                    sampled_incorrect.append(class_sample)
                                    remaining_samples -= class_sample_size
                            
                            # If we didn't fill our quota, sample randomly from the rest
                            if remaining_samples > 0 and len(incorrect_predictions) > sum(len(df) for df in sampled_incorrect):
                                already_sampled = pd.concat(sampled_incorrect).index if sampled_incorrect else pd.Index([])
                                remainder = incorrect_predictions.loc[~incorrect_predictions.index.isin(already_sampled)]
                                extra_samples = min(remaining_samples, len(remainder))
                                if extra_samples > 0:
                                    sampled_incorrect.append(remainder.sample(n=extra_samples, random_state=42))
                            
                            # Combine all samples
                            sampled_incorrect = pd.concat(sampled_incorrect) if sampled_incorrect else pd.DataFrame()
                            logger.debug(f"Sampled {len(sampled_incorrect):,} stratified incorrect predictions")
                        else:
                            # Simple random sample if groupby failed
                            sampled_incorrect = incorrect_predictions.sample(n=incorrect_sample_size, random_state=42)
                    else:
                        # Simple random sample if no 'actual' column
                        sampled_incorrect = incorrect_predictions.sample(n=incorrect_sample_size, random_state=42)
                except Exception as e:
                    logger.debug(f"Error in stratified sampling: {str(e)}, falling back to random sampling")
                    # Fallback to simple random sample
                    sampled_incorrect = incorrect_predictions.sample(n=incorrect_sample_size, random_state=42) if len(incorrect_predictions) > 0 else pd.DataFrame()
            else:
                sampled_incorrect = pd.DataFrame()
            
            # Sample from correct predictions
            if len(correct_predictions) > 0:
                if correct_sample_size > 0:
                    sampled_correct = correct_predictions.sample(n=correct_sample_size, random_state=42)
                else:
                    sampled_correct = pd.DataFrame()
            else:
                sampled_correct = pd.DataFrame()
            
            # Combine samples
            sampled_predictions = pd.concat([sampled_incorrect, sampled_correct])
            
            logger.info(f"Storing {len(sampled_predictions):,} predictions in metadata (sampled from {len(prediction_df):,} total)")
        else:
            # Store all if under the limit
            sampled_predictions = prediction_df
            logger.info(f"Storing all {len(prediction_df):,} predictions in metadata")
        
        # Convert to list for storing in metadata
        predictions_list = []
        for idx, row in sampled_predictions.iterrows():
            pred_dict = {
                'actual': str(row['actual']),
                'predicted': str(row['predicted'])
            }
            
            # Add index if it's meaningful
            if hasattr(idx, 'strftime'):  # For datetime index
                pred_dict['index'] = idx.strftime('%Y-%m-%d %H:%M:%S')
            elif not isinstance(idx, int) or idx != sampled_predictions.index.get_loc(idx):  # For non-default indices
                pred_dict['index'] = str(idx)
                
            predictions_list.append(pred_dict)
        
        # Add to metadata
        self.metadata['best_model']['prediction_samples'] = predictions_list
        
        # Add all incorrect predictions (up to a limit)
        max_incorrect = min(20, len(incorrect_predictions))
        if max_incorrect > 0:
            logger.debug(f"Storing {max_incorrect} detailed incorrect predictions")
            incorrect_list = []
            for idx, row in incorrect_predictions.head(max_incorrect).iterrows():
                incorrect_dict = {
                    'actual': str(row['actual']),
                    'predicted': str(row['predicted'])
                }
                
                # Add index if meaningful
                if hasattr(idx, 'strftime'):  # For datetime index
                    incorrect_dict['index'] = idx.strftime('%Y-%m-%d %H:%M:%S')
                elif not isinstance(idx, int) or idx != incorrect_predictions.index.get_loc(idx):  # For non-default indices
                    incorrect_dict['index'] = str(idx)
                
                incorrect_list.append(incorrect_dict)
            
            self.metadata['best_model']['incorrect_predictions'] = incorrect_list
        
        store_time = (datetime.now() - store_start).total_seconds()
        logger.debug(f"Prediction storage completed in {store_time:.2f} seconds")