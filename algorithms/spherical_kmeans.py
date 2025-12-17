import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
import numba
from numba import njit


def _l2_normalize_rows(X, eps=1e-12):
    X = X.astype(np.float32, copy=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    X /= norms
    return X


def _kmeans_pp_init(X, n_clusters, random_state=None):
    rng = check_random_state(random_state)
    n_samples, d = X.shape
    centers = np.empty((n_clusters, d), dtype=np.float32)
    first = rng.randint(0, n_samples)
    centers[0] = X[first]
    best_sim = X.dot(centers[0])
    for i in range(1, n_clusters):
        probs = (1.0 - best_sim).clip(min=0.0)
        tot = probs.sum()
        if tot <= 0:
            idx = rng.randint(0, n_samples)
        else:
            r = rng.random_sample() * tot
            csum = np.cumsum(probs)
            idx = int(np.searchsorted(csum, r))
            if idx >= n_samples:
                idx = n_samples - 1
        centers[i] = X[idx]
        sim_to_new = X.dot(centers[i])
        best_sim = np.maximum(best_sim, sim_to_new)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    return centers


@njit
def _apply_recompute_mem_efficient_nb(sims_sub, idxs, labels, l, second_best):
    m, k = sims_sub.shape
    for t in range(m):
        i = idxs[t]
        # find top and second top
        best = -2.0
        best_j = -1
        second = -2.0
        for j in range(k):
            s = sims_sub[t, j]
            if s > best:
                second = best
                best = s
                best_j = j
            elif s > second:
                second = s
        labels[i] = best_j
        l[i] = best
        second_best[i] = second


@njit
def _need_recompute_mem_efficient_nb(l, second_best):
    n = l.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    for i in range(n):
        # if l < second_best, we need to recompute
        mask[i] = l[i] < second_best[i]
    return mask


class SphericalKMeans(BaseEstimator, ClusterMixin):
    """
    Spherical KMeans with exact bound updates (Eqs. 6-7) and chunked dot products.

    Source: https://arxiv.org/pdf/2107.04074

    Parameters
    ----------
    n_clusters, n_init, max_iter, tol, init, random_state, verbose : as before
    batch_size : int
        Number of rows of X processed per chunk when computing X.dot(centers.T).
    """

    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        init="k-means++",
        random_state=None,
        verbose=False,
        batch_size=2048,
    ):
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.batch_size = int(batch_size)

    def _init_centers(self, X, rng):
        if self.init == "k-means++":
            return _kmeans_pp_init(X, self.n_clusters, rng)
        elif self.init == "random":
            n = X.shape[0]
            idx = rng.choice(n, self.n_clusters, replace=False)
            c = X[idx].copy()
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            return c
        else:
            raise ValueError("init must be 'k-means++' or 'random'")

    def _compute_assignments_in_batches(self, X, centers, labels, l, u):
        """
        Fill `labels`, `l` and `u` by computing sims in chunks:
        - labels: int32 array (n,)
        - l: float32 array (n,)
        - u: float32 array (n, k)
        """
        n = X.shape[0]
        k = centers.shape[0]
        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(n, start + bs)
            sims = X[start:end].dot(centers.T)  # (end-start, k)
            local_argmax = sims.argmax(axis=1).astype(np.int32)
            labels[start:end] = local_argmax
            # l: top similarity
            l[start:end] = sims[np.arange(end - start), local_argmax]
            # u rows: store full sims chunk
            u[start:end, :] = sims

    def _compute_assignments_mem_efficient(self, X, centers, labels, l, second_best):
        """
        - labels: int32 array (n,)
        - l: float32 array (n,) - best similarity
        - second_best: float32 array (n,) - second best similarity
        """
        n = X.shape[0]
        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(n, start + bs)
            sims = X[start:end].dot(centers.T)  # (end-start, k)
            # Find top 2 for each row
            batch_size = end - start
            for i in range(batch_size):
                row = sims[i]
                # Find best and second best
                if row[0] > row[1]:
                    best_val, second_val = row[0], row[1]
                    best_idx = 0
                else:
                    best_val, second_val = row[1], row[0]
                    best_idx = 1

                for j in range(2, len(row)):
                    val = row[j]
                    if val > best_val:
                        second_val = best_val
                        best_val = val
                        best_idx = j
                    elif val > second_val:
                        second_val = val

                labels[start + i] = best_idx
                l[start + i] = best_val
                second_best[start + i] = second_val

    def _compute_sims_for_indices_in_batches(self, X, centers, idxs):
        """
        Return sims for X[idxs] . centers.T, computed in batches to avoid memory spikes.
        Returns a dense array shape (len(idxs), k) (assembled from chunks).
        """
        m = len(idxs)
        k = centers.shape[0]
        sims_out = np.empty((m, k), dtype=np.float32, order="C")
        bs = self.batch_size
        # process contiguous subranges of idxs in chunks of bs
        # If idxs are not contiguous, we still proceed by grouping chunks of idxs
        for i in range(0, m, bs):
            j = min(m, i + bs)
            slice_idx = idxs[i:j]
            sims_chunk = X[slice_idx].dot(centers.T)  # (j-i, k)
            sims_out[i:j, :] = sims_chunk.astype(np.float32)
        return sims_out

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32, order="C")
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        n_samples, d = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        X = _l2_normalize_rows(X)
        rng = check_random_state(self.random_state)

        return self._fit(X, rng)

    def _fit(self, X, rng):
        """
        Memory-efficient version that only stores best and second-best similarities
        instead of full (n_samples, k) matrix. Reduces memory from O(n*k) to O(n).
        """
        n_samples, d = X.shape

        best_inertia = None
        best_centers = None
        best_labels = None

        for init_no in range(self.n_init):
            seed = rng.randint(0, 2**31 - 1)
            rng_init = check_random_state(seed)
            centers = self._init_centers(X, rng_init)
            k = self.n_clusters

            # allocate labels, l (best sim), second_best (second best sim)
            labels = np.empty(n_samples, dtype=np.int32)
            l = np.empty(n_samples, dtype=np.float32)
            second_best = np.empty(n_samples, dtype=np.float32)

            # initial assignment
            self._compute_assignments_mem_efficient(X, centers, labels, l, second_best)
            centers_old = centers.copy()

            for it in range(self.max_iter):
                # recompute centers
                centers_new = np.zeros_like(centers)
                for j in range(k):
                    idx_j = np.where(labels == j)[0]
                    if idx_j.size == 0:
                        r = rng_init.randint(0, n_samples)
                        centers_new[j] = X[r]
                    else:
                        centers_new[j] = X[idx_j].sum(axis=0)

                centers_new /= (
                    np.linalg.norm(centers_new, axis=1, keepdims=True) + 1e-12
                )

                # p(j) = similarity between old and new centers
                p = np.sum(centers_old * centers_new, axis=1).astype(np.float32)
                p = np.clip(p, -1.0, 1.0)

                # Update bounds: approximate update for memory efficiency
                # l(i) gets updated as in the exact method
                # second_best gets a conservative update
                for i in range(n_samples):
                    ai = labels[i]
                    li = l[i]
                    pa = p[ai]

                    # Update l[i] using exact formula
                    t1 = li * pa
                    tmp = max(0.0, (1.0 - li * li) * (1.0 - pa * pa))
                    l_new = t1 - np.sqrt(tmp) if tmp > 0 else t1
                    l[i] = np.clip(l_new, -1.0, 1.0)

                    # Conservative update for second_best: use maximum p value for non-assigned clusters
                    max_p_other = (
                        np.max([p[j] for j in range(k) if j != ai]) if k > 1 else 0.0
                    )
                    sb = second_best[i]
                    t2 = sb * max_p_other
                    tmp2 = max(0.0, (1.0 - sb * sb) * (1.0 - max_p_other * max_p_other))
                    sb_new = t2 + np.sqrt(tmp2) if tmp2 > 0 else t2
                    second_best[i] = np.clip(sb_new, -1.0, 1.0)

                # Find points that need recomputation
                need_mask = _need_recompute_mem_efficient_nb(l, second_best)
                idxs = np.nonzero(need_mask)[0]

                if idxs.size > 0:
                    # Recompute for selected indices in batches
                    bs = self.batch_size
                    for i in range(0, idxs.size, bs):
                        j = min(idxs.size, i + bs)
                        chunk_idxs = idxs[i:j].astype(np.int64)
                        sims_chunk = X[chunk_idxs].dot(centers_new.T).astype(np.float32)
                        _apply_recompute_mem_efficient_nb(
                            sims_chunk, chunk_idxs, labels, l, second_best
                        )

                # Check convergence
                sim_centers = np.sum(centers_old * centers_new, axis=1)
                max_center_move = np.max(1.0 - sim_centers)

                if self.verbose:
                    print(
                        f"[MEM-EFFICIENT] init={init_no} it={it} "
                        f"max_center_move={max_center_move:.6e} recomputed={idxs.size}"
                    )

                centers_old = centers_new
                centers = centers_new

                if max_center_move <= self.tol:
                    break

            total_sim = float(np.sum(l))
            inertia = -total_sim
            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32, order="C")
        X = _l2_normalize_rows(X)
        n = X.shape[0]
        labels = np.empty(n, dtype=np.int32)
        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(n, start + bs)
            sims = X[start:end].dot(self.cluster_centers_.T)
            labels[start:end] = sims.argmax(axis=1).astype(np.int32)
        return labels

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
