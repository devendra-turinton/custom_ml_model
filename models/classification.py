
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional, Tuple

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import ml_utils
from training_pipeline import BasePipeline
logger = logging.getLogger(__name__)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    balanced_accuracy_score
)


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
        super().__init__(**kwargs)
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
        X_train, X_test, y_train, y_test = super().preprocess_data()
        
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
        
        
        classifiers = {k: v for k, v in all_classifiers.items() if k in enabled_models}
        logger.info(f"Training {len(classifiers)} models: {', '.join(classifiers.keys())}")
    
            
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

