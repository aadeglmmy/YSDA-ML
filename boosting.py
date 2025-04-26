import contextlib
import inspect
import json
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from scipy.special import softmax
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.tree import DecisionTreeRegressor


class MyAdaBoostClassifier:
    """
    Multiclass AdaBoost implementation with SAMME.R algorithm
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            base_estimator: tp.Type[sklearn.base.BaseEstimator],
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: count of estimators
        :param base_estimator: base estimator (practically tree classifier)
        :param seed: global seed
        :param kwargs: keyword arguments of base estimator
        """
        self.n_classes = None
        self.error_history = []  # this is to track model learning process
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = kwargs
        # deduce which keywords are used to set seed for an estimator (sklearn or own tree implementation)
        signature = inspect.signature(self.base_estimator.__init__)
        self.seed_keyword = None
        if 'seed' in signature.parameters:
            self.seed_keyword = 'seed'
        elif 'random_state' in signature.parameters:
            self.seed_keyword = 'random_state'
        self.estimators = []
        self.estimator_weights = []

    def create_new_estimator(
            self,
            seed: int
    ):
        """
        Create a new base estimator with proper seed and keywords
        :param seed: Random seed for reproducibility
        :return: Instantiated estimator
        """
        estimator = self.base_estimator(**self.base_estimator_kwargs)
        if self.seed_keyword:
            setattr(estimator, self.seed_keyword, seed)
        return estimator

    def get_new_weights(
            self,
            true_labels: np.ndarray,
            predictions: np.ndarray,
            weights: np.ndarray
    ):
        """
        Calculate new weights according to SAMME.R scheme
        :param true_labels: [n_samples]
        :param predictions: [n_samples, n_classes]
        :param weights:     [n_samples]
        :return: normalized weights for next estimator fitting
        """
        K = predictions.shape[1]
        y_true = np.full(predictions.shape, -1 / (K - 1))
        y_true[np.arange(true_labels.size), true_labels] = 1

        log_preds = np.log(np.clip(predictions, self.eps, 1 - self.eps))

        weight_updates = np.exp(-((K - 1) / K) * np.sum(y_true * log_preds, axis=1))

        new_weights = weights * weight_updates
        new_weights /= np.sum(new_weights)

        return new_weights

    @staticmethod
    def get_estimator_error(
            estimator: sklearn.base.BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray
    ):
        """
        Calculate weighted error of an estimator
        :param estimator: Base estimator
        :param X: Feature matrix [n_samples, n_features]
        :param y: True labels [n_samples]
        :param weights: Current sample weights [n_samples]
        :return: Weighted error
        """
        predictions = estimator.predict(X)
        incorrect = (predictions != y).astype(float)
        weighted_error = np.dot(weights, incorrect) / np.sum(weights)
        return weighted_error

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        Sequentially fit estimators with updated weights on each iteration
        :param X: Feature matrix [n_samples, n_features]
        :param y: True labels [n_samples]
        :return: self
        """
        self.n_classes = len(np.unique(y))
        weights = np.full(X.shape[0], 1 / X.shape[0])

        for seed in self.rng.choice(
                max(MyAdaBoostClassifier.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            estimator = self.create_new_estimator(seed)
            self.estimators.append(estimator)

            estimator.fit(X, y, sample_weight=weights)

            predictions = estimator.predict_proba(X)

            error = self.get_estimator_error(estimator, X, y, weights)
            self.error_history.append(error)

            if error >= 1 - self.eps:
                continue

            estimator_weight = (np.log((1 - error) / error) + np.log(self.n_classes - 1))
            self.estimator_weights.append(estimator_weight)

            weights = self.get_new_weights(y, predictions, weights)

        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        Predicts probability of each class
        :param X: [n_samples, n_features]
        :return: array of probabilities of shape [n_samples, n_classes]
        """
        proba_sum = np.zeros((X.shape[0], self.n_classes))

        for estimator, weight in zip(self.estimators, self.estimator_weights):
            proba = estimator.predict_proba(X)
            proba_sum += weight * np.log(np.clip(proba, self.eps, 1 - self.eps))

        probas = softmax(proba_sum, axis=1)
        return probas

    def predict(
            self,
            X: np.ndarray
    ):
        """
        Predicts class labels (based on predicted class probabilities)
        :param X: [n_samples, n_features]
        :return: array of class predictions of shape [n_samples]
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return predictions


class MyBinaryTreeGradientBoostingClassifier:
    """
    *Binary* gradient boosting with trees using
    negative log-likelihood loss with constant learning rate.
    Trees are to predict logits.
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: estimators count
        :param learning_rate: hard learning rate
        :param seed: global seed
        :param kwargs: kwargs of base estimator which is sklearn TreeRegressor
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_logits = None
        self.rng = np.random.default_rng(seed)
        self.base_estimator = DecisionTreeRegressor
        self.base_estimator_kwargs = kwargs
        self.estimators = []
        self.loss_history = []  # this is to track model learning process

    def create_new_estimator(self, seed):
        return self.base_estimator(**self.base_estimator_kwargs, random_state=seed)

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ) -> float:
        """
        Compute negative log-likelihood for logits,
        use clipping for logarithms with self.eps.
        This is used to track model learning process
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return: loss value
        """
        probabilities = 1 / (1 + np.exp(-logits))
        probabilities = np.clip(probabilities, MyBinaryTreeGradientBoostingClassifier.eps,
                                1 - MyBinaryTreeGradientBoostingClassifier.eps)
        nll = -np.sum(
            true_labels * np.log(probabilities) +
            (1 - true_labels) * np.log(1 - probabilities)
        )
        return nll

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of log-likelihood w.r.t logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions.
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return: gradient array
        """
        probabilities = 1 / (1 + np.exp(-logits))
        probabilities = np.clip(probabilities, MyBinaryTreeGradientBoostingClassifier.eps,
                                1 - MyBinaryTreeGradientBoostingClassifier.eps)
        gradient = probabilities - true_labels
        return gradient

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        sequentially fit estimators to reduce residual on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        self.loss_history = []
        # only should be fitted on datasets with binary target
        assert (np.unique(y) == np.arange(2)).all()
        # init predictions with mean target (mind that these are logits!)
        mean_target = np.mean(y)
        self.initial_logits = np.log(mean_target / (1 - mean_target))
        # create starting logits
        logits = np.full(y.shape, self.initial_logits)
        # init loss history with starting negative log-likelihood
        self.loss_history.append(self.cross_entropy_loss(y, logits))
        # sequentially fit estimators with random seeds
        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            # add newly created estimator
            estimator = self.create_new_estimator(seed)
            self.estimators.append(estimator)
            # compute gradient
            gradient = self.cross_entropy_loss_gradient(y, logits)
            # fit estimator on gradient residual
            estimator.fit(X, -gradient)
            # adjust logits with learning rate
            logits += self.learning_rate * estimator.predict(X)
            # append new loss to history
            self.loss_history.append(self.cross_entropy_loss(y, logits))
        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        :param X: [n_samples]
        :return:
        """
        # init logits using precalculated values
        logits = np.full(X.shape[0], self.initial_logits)
        # sequentially adjust logits with learning rate
        for estimator in self.estimators:
            logits += self.learning_rate * estimator.predict(X)
        # don't forget to convert logits to probabilities
        probabilities = 1 / (1 + np.exp(-logits))
        probabilities = np.clip(probabilities, self.eps, 1 - self.eps)
        return probabilities

    def predict(
            self,
            X: np.ndarray
    ):
        """
        calculate predictions using predict_proba
        :param X: [n_samples]
        :return:
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


class Logger:
    """Logger performs data management and stores scores and other relevant information"""

    def __init__(self, logs_path: tp.Union[str, os.PathLike]):
        self.path = pathlib.Path(logs_path)

        records = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith('.json'):
                    uuid = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            logged_data = json.load(f)
                            records.append(
                                {
                                    'id': uuid,
                                    **logged_data
                                }
                            )
                        except json.JSONDecodeError:
                            pass
        if records:
            self.leaderboard = pd.DataFrame.from_records(records, index='id')
        else:
            self.leaderboard = pd.DataFrame(index=pd.Index([], name='id'))

        self._current_run = None

    class Run:
        """Run incapsulates information for a particular entry of logged material. Each run is solitary experiment"""

        def __init__(self, name, storage, path):
            self.name = name
            self._storage = storage
            self._path = path
            self._storage.append(pd.Series(name=name))

        def log(self, key, value):
            self._storage.loc[self.name, key] = value

        def log_values(self, log_values: tp.Dict[str, tp.Any]):
            for key, value in log_values.items():
                self.log(key, value)

        def save_logs(self):
            with open(self._path / f'{self.name}.json', 'w+') as f:
                json.dump(self._storage.loc[self.name].to_dict(), f)

        def log_artifact(self, fname: str, writer: tp.Callable):
            with open(self._path / fname, 'wb+') as f:
                writer(f)

    @contextlib.contextmanager
    def run(self, name: tp.Optional[str] = None):
        if name is None:
            name = str(uuid.uuid4())
        # elif name in self.leaderboard.index:
        #     raise NameError("Run with given name already exists, name should be unique")
        else:
            name = name.replace(' ', '_')
        self._current_run = Logger.Run(name, self.leaderboard, self.path / name)
        os.makedirs(self.path / name, exist_ok=True)
        try:
            yield self._current_run
        finally:
            self._current_run.save_logs()


def load_predictions_dataframe(filename, column_prefix, index):
    with open(filename, 'rb') as file:
        data = np.load(file)
        dataframe = pd.DataFrame(data, columns=[f'{column_prefix}_{i}' for i in range(data.shape[1])],
                                 index=index)
        return dataframe


class ExperimentHandler:
    """This class perfoms experiments with given model, measures metrics and logs everything for thorough comparison"""
    stacking_prediction_filename = 'cv_stacking_prediction.npy'
    test_stacking_prediction_filename = 'test_stacking_prediction.npy'

    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            cv_iterable: tp.Iterable,
            logger: Logger,
            metrics: tp.Dict[str, tp.Union[tp.Callable, str]],
            n_jobs=-1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._cv_iterable = cv_iterable
        self.logger = logger
        self._metrics = metrics
        self._n_jobs = n_jobs

    def score_test(self, estimator, metrics, run, test_data=None):
        """
        Computes scores for test data and logs them to given run
        :param estimator: fitted estimator
        :param metrics: metrics to compute
        :param run: run to log into
        :param test_data: optional argument if one wants to pass augmented test dataset
        :return: None
        """
        if test_data is None:
            test_data = self.X_test
        test_scores = _score(estimator, test_data, self.y_test, metrics)
        run.log_values({key + '_test': value for key, value in test_scores.items()})

    def score_cv(self, estimator, metrics, run):
        """
        computes scores on cross-validation
        :param estimator: estimator to fit
        :param metrics: metrics to compute
        :param run: run to log to
        :return: None
        """
        cross_val_results = sklearn.model_selection.cross_validate(
            estimator,
            self.X_train,
            self.y_train,
            cv=self._cv_iterable,
            n_jobs=self._n_jobs,
            scoring=metrics
        )
        for key, value in cross_val_results.items():
            if key.startswith('test_'):
                metric_name = key.split('_', maxsplit=1)[1]
                mean_score = np.mean(value)
                std_score = np.std(value)
                run.log_values(
                    {
                        metric_name + '_mean': mean_score,
                        metric_name + '_std': std_score
                    }
                )

    def generate_stacking_predictions(self, estimator, run):
        """
        generates predictions over cross-validation folds, then saves them as artifacts
        returns fitted estimator for convinience and les train overhead
        :param estimator: estimator to use
        :param run: run to log to
        :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
        """
        if hasattr(estimator, "predict_proba"):
            method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"
        cross_val_stacking_prediction = sklearn.model_selection.cross_val_predict(
            estimator,
            self.X_train,
            self.y_train,
            cv=self._cv_iterable,
            n_jobs=self._n_jobs,
            method=method
        )
        run.log_artifact(ExperimentHandler.stacking_prediction_filename,
                         lambda file: np.save(file, cross_val_stacking_prediction))
        estimator.fit(self.X_train, self.y_train)
        test_stacking_prediction = getattr(estimator, method)(self.X_test)
        run.log_artifact(ExperimentHandler.test_stacking_prediction_filename,
                         lambda file: np.save(file, test_stacking_prediction))
        return estimator, cross_val_stacking_prediction, test_stacking_prediction

    def get_metrics(self, estimator):
        """
        get callable metrics with estimator validation
        (e.g. estimator has predict_proba necessary for likelihood computation, etc)
        """
        return _check_multimetric_scoring(estimator, self._metrics)

    def run(self, estimator: sklearn.base.BaseEstimator, name=None):
        """
        perform run for given estimator
        :param estimator: estimator to use
        :param name: name of run for convinience and consitent logging
        :return: leaderboard with conducted run
        """
        metrics = self.get_metrics(estimator)
        with self.logger.run(name=name) as run:
            # compute predictions over cross-validation
            self.score_cv(estimator, metrics, run)
            fitted_on_train, _, _ = self.generate_stacking_predictions(estimator, run)
            self.score_test(fitted_on_train, metrics, run, test_data=self.X_test)
            return self.logger.leaderboard.loc[[run.name]]

    def get_stacking_predictions(self, run_names):
        """
        :param run_names: run names for which to extract stacking predictions for averaging and stacking
        :return: dataframe with predictions indexed by run names
        """
        train_dataframes = []
        test_dataframes = []
        for run_name in run_names:
            train_filename = self.logger.path / run_name / ExperimentHandler.stacking_prediction_filename
            train_dataframes.append(load_predictions_dataframe(filename=train_filename, column_prefix=run_name,
                                                               index=self.X_train.index))
            test_filename = self.logger.path / run_name / ExperimentHandler.test_stacking_prediction_filename
            test_dataframes.append(load_predictions_dataframe(filename=test_filename, column_prefix=run_name,
                                                              index=self.X_test.index))

        return pd.concat(train_dataframes, axis=1), pd.concat(test_dataframes, axis=1)
