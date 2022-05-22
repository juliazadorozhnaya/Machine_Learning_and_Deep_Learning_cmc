import numpy as np



def prod_non_zero_diag(X: np.ndarray) -> int:
    X = np.diag(X)
    if len(X) == len(X[X == 0]):
        return -1
    return np.prod(X[X != 0])


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.all(np.sort(x) == np.sort(y))


def max_after_zero(x: np.ndarray) -> int:
    x = np.append(x, -1)
    indx = np.where(x == 0)[0] + 1
    indx = indx[np.where(indx < len(x))]
    if len(indx) == 0:
        return -1
    return max(x[indx])


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(image, weights)


def run_length_encoding(x: np.ndarray) -> (np.ndarray, np.ndarray):
    d = np.diff(x)
    a = (np.array(np.where(d != 0)) + 1).flatten()
    if a.size != 0:
        b = np.insert(np.diff(a), 0, a[0])
        return np.append(x[np.where(d != 0)], x[x.size - 1]), \
               np.append(b, d.size - a[a.size - 1] + 1)
    else:
        return np.array([x[0]]), np.array([x.size])


def pairwise_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, :, None] - Y[:, :, None].T, axis=1)