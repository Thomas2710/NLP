from sklearn.model_selection import cross_validate


def run_experiment(params, experiment_id, all_data):
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    vectorizer = params["vectorizer"]
    clf = params["clf"]
    split = params["split"]
    vector = vectorizer.fit_transform(all_data.data)
    scores = cross_validate(clf, vector, all_data.target, cv=split, scoring=scoring)
    print(
        ""
        + experiment_id
        + " accuracy : {:.3}".format(
            sum(scores["test_accuracy"]) / len(scores["test_accuracy"])
        )
    )
    print(
        ""
        + experiment_id
        + " precision_macro : {:.3}".format(
            sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])
        )
    )
    print(
        ""
        + experiment_id
        + " recall_macro : {:.3}".format(
            sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"])
        )
    )
    print("")
