"""
ifcm.py

Improved Fuzzy C-Means (IFCM) segmentation for lung CT preprocessing.

This module implements a spatially regularized fuzzy clustering method for
initial emphysema-aware tissue separation before TransUNet segmentation.

Core idea:
- Standard FCM clusters pixels by intensity similarity.
- IFCM additionally uses local spatial context to reduce noise sensitivity.
- The output can be used as a structured prior for downstream deep models.

The implementation is intentionally compact and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class IFCMConfig:
    clusters: int = 3
    fuzziness: float = 2.0
    spatial_weight: float = 0.30
    max_iterations: int = 100
    tolerance: float = 1e-4
    neighborhood_size: int = 3
    random_state: int = 42


class ImprovedFuzzyCMeans:
    """
    Spatially regularized Improved Fuzzy C-Means for grayscale CT images.

    Parameters
    ----------
    clusters:
        Number of clusters.
    fuzziness:
        Fuzziness coefficient m. Must be greater than 1.
    spatial_weight:
        Weight for local spatial regularization.
    max_iterations:
        Maximum number of update iterations.
    tolerance:
        Convergence threshold for centroid movement.
    neighborhood_size:
        Kernel size used for local spatial averaging.
    random_state:
        Random seed for reproducible initialization.
    """

    def __init__(
        self,
        clusters: int = 3,
        fuzziness: float = 2.0,
        spatial_weight: float = 0.30,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        neighborhood_size: int = 3,
        random_state: int = 42,
    ) -> None:
        if clusters < 2:
            raise ValueError("clusters must be >= 2.")
        if fuzziness <= 1:
            raise ValueError("fuzziness must be > 1.")

        self.clusters = int(clusters)
        self.fuzziness = float(fuzziness)
        self.spatial_weight = float(spatial_weight)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.neighborhood_size = int(neighborhood_size)
        self.random_state = int(random_state)

        self.centroids_: Optional[np.ndarray] = None
        self.membership_: Optional[np.ndarray] = None
        self.objective_history_: list[float] = []

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        minimum = float(image.min())
        maximum = float(image.max())

        if maximum - minimum < 1e-8:
            return np.zeros_like(image, dtype=np.float32)

        return ((image - minimum) / (maximum - minimum)).astype(np.float32)

    def _initialize_centroids(self, pixels: np.ndarray) -> np.ndarray:
        """
        Histogram-informed centroid initialization using intensity quantiles.
        """
        quantiles = np.linspace(0.05, 0.95, self.clusters)
        centroids = np.quantile(pixels, quantiles).astype(np.float32)

        if len(np.unique(np.round(centroids, 6))) < self.clusters:
            rng = np.random.default_rng(self.random_state)
            centroids = rng.uniform(
                low=float(pixels.min()),
                high=float(pixels.max()) + 1e-6,
                size=self.clusters,
            ).astype(np.float32)

        return centroids

    def _compute_distance(self, pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute absolute intensity distance between pixels and centroids.
        """
        distance = np.abs(pixels[:, None] - centroids[None, :])
        distance = np.maximum(distance, 1e-8)
        return distance.astype(np.float32)

    def _update_membership(self, distance: np.ndarray) -> np.ndarray:
        """
        Standard FCM membership update.
        """
        power = 2.0 / (self.fuzziness - 1.0)

        ratio = distance[:, :, None] / distance[:, None, :]
        ratio = np.maximum(ratio, 1e-8)

        denominator = np.sum(ratio ** power, axis=2)
        membership = 1.0 / np.maximum(denominator, 1e-8)

        return membership.astype(np.float32)

    def _spatial_regularization(
        self,
        membership: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Smooth membership maps using local spatial context.

        This reduces isolated noisy assignments and encourages coherent lung
        tissue regions.
        """
        h, w = image_shape
        regularized = np.zeros_like(membership, dtype=np.float32)

        kernel_size = self.neighborhood_size
        if kernel_size % 2 == 0:
            kernel_size += 1

        for cluster_id in range(self.clusters):
            member_map = membership[:, cluster_id].reshape(h, w)
            local_mean = cv2.blur(member_map, (kernel_size, kernel_size))
            regularized[:, cluster_id] = local_mean.reshape(-1)

        combined = (1.0 - self.spatial_weight) * membership + self.spatial_weight * regularized
        combined = combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-8)

        return combined.astype(np.float32)

    def _update_centroids(self, pixels: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """
        Update cluster centroids using fuzzy memberships.
        """
        um = membership ** self.fuzziness
        numerator = np.sum(um * pixels[:, None], axis=0)
        denominator = np.sum(um, axis=0)
        centroids = numerator / np.maximum(denominator, 1e-8)
        return centroids.astype(np.float32)

    def _objective(
        self,
        pixels: np.ndarray,
        membership: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """
        Compute FCM objective value.
        """
        distance_sq = (pixels[:, None] - centroids[None, :]) ** 2
        obj = np.sum((membership ** self.fuzziness) * distance_sq)
        return float(obj)

    def fit(self, image: np.ndarray) -> "ImprovedFuzzyCMeans":
        """
        Fit IFCM to a single grayscale image.
        """
        image = self._normalize(image)
        h, w = image.shape

        pixels = image.reshape(-1).astype(np.float32)
        centroids = self._initialize_centroids(pixels)

        self.objective_history_ = []

        for _ in range(self.max_iterations):
            previous_centroids = centroids.copy()

            distance = self._compute_distance(pixels, centroids)
            membership = self._update_membership(distance)
            membership = self._spatial_regularization(membership, (h, w))
            centroids = self._update_centroids(pixels, membership)

            objective_value = self._objective(pixels, membership, centroids)
            self.objective_history_.append(objective_value)

            centroid_shift = np.linalg.norm(centroids - previous_centroids)
            if centroid_shift < self.tolerance:
                break

        order = np.argsort(centroids)
        centroids = centroids[order]
        membership = membership[:, order]

        self.centroids_ = centroids.astype(np.float32)
        self.membership_ = membership.astype(np.float32)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict hard cluster labels for a grayscale image.
        """
        if self.centroids_ is None:
            raise RuntimeError("IFCM must be fitted before calling predict().")

        image = self._normalize(image)
        h, w = image.shape
        pixels = image.reshape(-1).astype(np.float32)

        distance = self._compute_distance(pixels, self.centroids_)
        membership = self._update_membership(distance)
        membership = self._spatial_regularization(membership, (h, w))

        labels = np.argmax(membership, axis=1).reshape(h, w)
        return labels.astype(np.uint8)

    def fit_predict(self, image: np.ndarray) -> np.ndarray:
        """
        Fit IFCM and return hard cluster labels.
        """
        self.fit(image)

        if self.membership_ is None:
            raise RuntimeError("Membership was not computed.")

        h, w = image.shape
        labels = np.argmax(self.membership_, axis=1).reshape(h, w)
        return labels.astype(np.uint8)

    def membership_maps(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Return membership maps with shape [C, H, W].
        """
        if self.membership_ is None:
            raise RuntimeError("IFCM must be fitted before accessing membership maps.")

        h, w = image_shape
        maps = self.membership_.T.reshape(self.clusters, h, w)
        return maps.astype(np.float32)

    def emphysema_prior_mask(
        self,
        image_shape: Tuple[int, int],
        cluster_index: Optional[int] = None,
        threshold: float = 0.5,
        cleanup: bool = True,
    ) -> np.ndarray:
        """
        Generate a binary emphysema prior mask.

        In normalized CT slices, emphysematous regions commonly appear as
        lower-density zones. By default, the lowest-intensity cluster is used.
        """
        if self.membership_ is None:
            raise RuntimeError("IFCM must be fitted before creating a prior mask.")

        h, w = image_shape

        if cluster_index is None:
            if self.centroids_ is None:
                cluster_index = 0
            else:
                cluster_index = int(np.argmin(self.centroids_))

        prior = self.membership_[:, cluster_index].reshape(h, w)
        mask = (prior >= threshold).astype(np.uint8)

        if cleanup:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask.astype(np.float32)


def run_ifcm(
    image: np.ndarray,
    clusters: int = 3,
    fuzziness: float = 2.0,
    spatial_weight: float = 0.30,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    neighborhood_size: int = 3,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Convenience function for running IFCM on one image.

    Returns
    -------
    dict
        {
            "labels": hard cluster label map,
            "prior_mask": binary emphysema prior,
            "membership": membership maps [C, H, W],
            "centroids": cluster centroids,
            "objective_history": objective curve
        }
    """
    model = ImprovedFuzzyCMeans(
        clusters=clusters,
        fuzziness=fuzziness,
        spatial_weight=spatial_weight,
        max_iterations=max_iterations,
        tolerance=tolerance,
        neighborhood_size=neighborhood_size,
        random_state=random_state,
    )

    labels = model.fit_predict(image)
    membership = model.membership_maps(image.shape)
    prior_mask = model.emphysema_prior_mask(image.shape)

    return {
        "labels": labels,
        "prior_mask": prior_mask,
        "membership": membership,
        "centroids": model.centroids_,
        "objective_history": np.asarray(model.objective_history_, dtype=np.float32),
    }


def run_ifcm_from_config(image: np.ndarray, config: Dict) -> Dict[str, np.ndarray]:
    """
    Run IFCM using config.yaml parameters.
    """
    cfg = config.get("ifcm", {})

    return run_ifcm(
        image=image,
        clusters=int(cfg.get("clusters", 3)),
        fuzziness=float(cfg.get("fuzziness", 2.0)),
        spatial_weight=float(cfg.get("spatial_weight", 0.30)),
        max_iterations=int(cfg.get("max_iterations", 100)),
        tolerance=float(cfg.get("tolerance", 1e-4)),
        neighborhood_size=int(cfg.get("neighborhood_size", 3)),
        random_state=int(config.get("project", {}).get("seed", 42)),
    )


def append_ifcm_channel(image: np.ndarray, prior_mask: np.ndarray) -> np.ndarray:
    """
    Create a two-channel input: [CT image, IFCM prior mask].

    This can be used when TransUNet is configured to consume IFCM-guided input.
    """
    image = image.astype(np.float32)
    prior_mask = prior_mask.astype(np.float32)

    if image.ndim != 2:
        raise ValueError("image must have shape [H, W].")

    if prior_mask.shape != image.shape:
        prior_mask = cv2.resize(
            prior_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    stacked = np.stack([image, prior_mask], axis=0)
    return stacked.astype(np.float32)


def batch_ifcm_prior(
    images: np.ndarray,
    config: Optional[Dict] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate IFCM prior masks for a batch of images.

    Parameters
    ----------
    images:
        NumPy array with shape [B, H, W] or [B, 1, H, W].
    config:
        Optional config dictionary.

    Returns
    -------
    np.ndarray
        Prior masks with shape [B, 1, H, W].
    """
    if images.ndim == 4:
        images_2d = images[:, 0]
    elif images.ndim == 3:
        images_2d = images
    else:
        raise ValueError("images must have shape [B, H, W] or [B, 1, H, W].")

    masks = []

    for image in images_2d:
        if config is not None:
            result = run_ifcm_from_config(image, config)
        else:
            result = run_ifcm(image, **kwargs)

        masks.append(result["prior_mask"])

    masks = np.stack(masks, axis=0)
    masks = np.expand_dims(masks, axis=1)

    return masks.astype(np.float32)
