#Model Selection: Grid Search Cross Validation

from sklearn.model_selection import GridSearchCV

def grid_search_cv(estimator,param_grid,scoring=None,**kwargs):
    """
    Exhaustive search over specified parameter values for an estimator.

    Parameters : 
        estimator : estimator object.
            This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.
        
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of parameter settings.
        
        scoring : string, callable, list/tuple, dict or None, default: None
            A single string (see The scoring parameter: defining model evaluation rules) or a callable (see Defining your scoring strategy from metric functions) to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique) strings or a dict with names as keys and callables as values.

            NOTE that when using custom scorers, each scorer should return a single value. Metric functions returning a list/array of values can be wrapped into multiple scorers that return one value each.

            See Specifying multiple metrics for evaluation for an example.

            If None, the estimator’s score method is used.
        
        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
        
        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. This parameter can be:

                    None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
                    An int, giving the exact number of total jobs that are spawned
                    A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’

        iid : boolean, default=’warn’
            If True, return the average score across folds, weighted by the number of samples in each test set. In this case, the data is assumed to be identically distributed across the folds, and the loss minimized is the total loss per sample, and not the mean loss across the folds. If False, return the average score across folds. Default is True, but will change to False in version 0.22, to correspond to the standard definition of cross-validation.
        
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

            Refer User Guide for the various cross-validation strategies that can be used here.
        
        refit : boolean, string, or callable, default=True

            Refit an estimator using the best found parameters on the whole dataset.

            For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.

            Where there are considerations other than maximum score in choosing a best estimator, refit can be set to a function which returns the selected best_index_ given cv_results_.

            The refitted estimator is made available at the best_estimator_ attribute and permits using predict directly on this GridSearchCV instance.

            Also for multiple metric evaluation, the attributes best_index_, best_score_ and best_params_ will only be available if refit is set and all of them will be determined w.r.t this specific scorer. best_score_ is not returned if refit is callable.

            See scoring parameter to know more about multiple metric evaluation.

            Changed in version 0.20: Support for callable added.
        
        verbose : integer
            Controls the verbosity: the higher, the more messages.
        
        error_score : ‘raise’ or numeric
            Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which will always raise the error. Default is ‘raise’ but from version 0.22 it will change to np.nan.
        
        return_train_score : boolean, default=False
            If False, the cv_results_ attribute will not include training scores. Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
    """
    return GridSearchCV(estimator,param_grid,scoring,**kwargs)