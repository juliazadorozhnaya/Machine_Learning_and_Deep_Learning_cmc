import numpy as np

import sklearn
import sklearn.metrics
from sklearn.cluster import KMeans


def silhouette_score(x, labels):

    if len(np.unique(labels)) == 1:
        return 0

    s = np.zeros(len(x))
    clusters = []

    for c in np.unique(labels):

        if len((labels == c)) == 1:
            s[(labels == c)] = 0
            continue

        values = np.sum(
            sklearn.metrics.pairwise_distances(x, x)[:, (labels == c)], axis=1
        )
        tmp = values / (labels == c).sum()
        tmp[(labels == c)] = sklearn.metrics.pairwise_distances(x, x).max() + 1
        s[(labels == c) * ((labels == c).sum() > 1)] = values[
            (labels == c) * ((labels == c).sum() > 1)
        ] / ((labels == c).sum() - 1)
        clusters.append(tmp)

    d = np.min(np.stack(clusters), axis=0)
    sil = np.zeros(len(x))
    mask = (d != 0) * (s != 0)

    sil[mask] = (d - s)[mask] / np.maximum(d, s)[mask]
    sil_score = sil.mean(axis=0)
    return sil_score


def bcubed_score(true_labels, predicted_labels):

    mask1 = (
        np.broadcast_to(true_labels, (true_labels.shape[0], true_labels.shape[0])) -
        np.broadcast_to(true_labels, (true_labels.shape[0], true_labels.shape[0])).T
    ) == 0
    mask2 = (
        np.broadcast_to(
            predicted_labels, (predicted_labels.shape[0], predicted_labels.shape[0])
        ) -
        np.broadcast_to(
            predicted_labels, (predicted_labels.shape[0], predicted_labels.shape[0])
        ).T
    ) == 0

    score = (
        2 *
        (
            np.mean(np.sum(mask1 * mask2, axis=0) / np.sum(mask1, axis=0)) *
            np.mean(np.sum(mask1 * mask2, axis=0) / np.sum(mask2, axis=0))
        ) /
        (
            np.mean(np.sum(mask1 * mask2, axis=0) / np.sum(mask1, axis=0)) +
            np.mean(np.sum(mask1 * mask2, axis=0) / np.sum(mask2, axis=0))
        )
    )
    return score


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, data, labels):
        self.classifier = KMeans(n_clusters=self.n_clusters)
        self.classifier.fit(data)
        return self

    def predict(self, data):
        return

    def _best_fit_classification(self, cluster_labels, true_labels):
        n_classes = self.n_clusters
        mask = true_labels >= 0
        clusters = np.split(
            cluster_labels.argsort(),
            np.unique(cluster_labels, return_counts=True)[1].cumsum(),
        )[:-1]
        mapping = []
        for cl in clusters:
            lbl, lbl_count = np.unique(true_labels[cl], return_counts=True)

            if len(lbl) == 1 and lbl[0] == -1:
                lbl, lbl_count = np.unique(true_labels[mask], return_counts=True)
                mapping.append(lbl[np.argmax(lbl_count)])
            else:
                lbl, lbl_count = lbl[lbl >= 0], lbl_count[lbl >= 0]
                mapping.append(lbl[np.argmax(lbl_count[lbl >= 0])])

        while len(mapping) < n_classes:
            lbl, lbl_count = np.unique(true_labels[mask], return_counts=True)
            mapping.append(lbl[np.argmax(lbl_count)])

        mapping = np.array(mapping)
        predicted_labels = mapping[cluster_labels]
        return mapping, predicted_labels
