import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.transform import AffineTransform


class Mosaic:
    def __init__(self):
        self.images_list = []
        self.translations_list = []
        self.transformations_list = []

        self.mosaic_size = (0, 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mosaic):
            return False

        if tuple(self.mosaic_size) != tuple(other.mosaic_size):
            return False

        if self.n_images() != other.n_images():
            return False

        for image, other_image in zip(self.images_list, other.images_list):
            if not np.allclose(image, other_image):
                return False

        for mosaic_loc, other_loc in zip(
            self.translations_list, other.translations_list
        ):
            if not np.allclose(mosaic_loc.params, other_loc.params):
                return False

        for mosaic_loc, other_loc in zip(
            self.transformations_list, other.transformations_list
        ):
            if not np.allclose(mosaic_loc.params, other_loc.params):
                return False

        return True

    def copy(self) -> "Mosaic":
        new_mosaico = Mosaic()

        new_mosaico.images_list = deepcopy(self.images_list)
        new_mosaico.translations_list = deepcopy(self.translations_list)
        new_mosaico.transformations_list = deepcopy(self.transformations_list)
        new_mosaico.mosaic_size = self.mosaic_size

        return new_mosaico

    def add(self, image, translation=None, tr_format="ij"):
        """
        location: representando la translación a realizar (No centrada)
        tr_format: 'ij':(y,x); 'xy': (x y)
        """
        if translation is None:
            translation = [0, 0]
        if tr_format == "ij":
            translation = translation[::-1]

        assert len(translation) == 2 and tr_format in ["ij", "xy"]

        tf = AffineTransform(translation=translation)
        self._update_and_add(image, tf)

    def n_images(self):
        return len(self.images_list)

    def _tf_point(self, xy_point, tf):
        point = np.array([*xy_point, 1], "float32")
        point_tf = tf.params @ point
        return point_tf[:2]

    def _tf_corners(self, image, tf):
        h, w = image.shape[:2]

        top_left = self._tf_point([0, 0], tf)
        top_right = self._tf_point([w, 0], tf)
        bottom_left = self._tf_point([0, h], tf)
        bottom_right = self._tf_point([w, h], tf)

        return top_left, top_right, bottom_left, bottom_right

    def _get_centered_transform(self, translation, transformation, image_shape):
        combinated = translation.params @ transformation.params
        return AffineTransform(to_centered_transform(combinated, image_shape))

    def image_centered_transform(self, idx: int) -> AffineTransform:
        translation = self.translations_list[idx]
        transformation = self.transformations_list[idx]
        image_shape = self.images_list[idx].shape[:2]
        return self._get_centered_transform(translation, transformation, image_shape)

    def _round_mosaic_size(self, size: Tuple[float, float]) -> Tuple[int, int]:
        return tuple(np.ceil(size).astype(int))

    def _update_and_add(self, new_image, new_image_tf):
        h, w = self.mosaic_size
        new_w, new_h = w, h
        corners = self._tf_corners(new_image, new_image_tf)

        # 1. ¿Se sale por la izquierda la nueva imagen?
        if np.min(corners, axis=0)[0] < 0:
            tmp = np.min(corners, axis=0)[0]
            x_shift = abs(tmp)
            new_w += abs(tmp)
        else:
            x_shift = 0

        # 2. ¿Se sale por la derecha la nueva imagen?
        if np.max(corners, axis=0)[0] > w:
            new_w += np.max(corners, axis=0)[0] - w

        # 3. ¿Se sale por la arriba la nueva imagen?
        if np.min(corners, axis=0)[1] < 0:
            tmp = np.min(corners, axis=0)[1]
            y_shift = abs(tmp)
            new_h += abs(tmp)
        else:
            y_shift = 0

        # 4. ¿Se sale por la abajo la nueva imagen?
        if np.max(corners, axis=0)[1] > h:
            new_h += np.max(corners, axis=0)[1] - h

        self.mosaic_size = self._round_mosaic_size((new_h, new_w))
        self.images_list.append(new_image)
        self.translations_list.append(new_image_tf)
        self.transformations_list.append(AffineTransform())

        translations_updated = []
        shift = np.zeros((3, 3))
        shift[:2, -1] = [x_shift, y_shift]
        for tf in self.translations_list:
            tf_updated = AffineTransform(tf.params + shift)
            translations_updated.append(tf_updated)
        self.translations_list = translations_updated

    def image(self, pad=0, reverse_order=True, rgb=False) -> Optional[np.ndarray]:
        images_order = np.arange(len(self.images_list))
        if reverse_order:
            images_order = images_order[::-1]

        return self.image_with_order(images_order, pad, rgb)

    def image_with_order(self, images_order, pad=0, rgb=False) -> Optional[np.ndarray]:
        """From background to foreground"""
        if len(self.images_list) == 0:
            return None

        if len(images_order) == 0:
            raise ValueError("images order can not be empty.")

        shift = np.zeros((3, 3))
        shift[:2, -1] = [pad, pad]
        h, w = self.mosaic_size

        mosaico = np.pad(np.zeros((h + pad * 2, w + pad * 2)), pad)
        if rgb:
            mosaico = np.dstack((mosaico, mosaico, mosaico))

        mosaico_size = mosaico.shape[:2][::-1]
        for idx in images_order:
            image, translation, transformation = self.get_image_data(idx)
            translation_padded = AffineTransform(translation.params + shift)

            centered_tf = self._get_centered_transform(
                translation_padded, transformation, image.shape[:2]
            )
            indices = cv2.warpPerspective(
                np.ones_like(image), centered_tf.params, mosaico_size
            )
            if rgb and image.ndim == 2:
                image = np.dstack((image, image, image))

            image_tf = cv2.warpPerspective(image, centered_tf.params, mosaico_size)
            mosaico = np.where(indices > 0, image_tf, mosaico)

        return mosaico

    def mask(self, idx, only_translation=True):
        image_1, tl_1, tf_1 = self.get_image_data(idx)
        if only_translation:
            tf_1_centered = tl_1
        else:
            tf_1_centered = self._get_centered_transform(tl_1, tf_1, image_1.shape[:2])
        mask_image_1 = cv2.warpPerspective(
            np.ones_like(image_1), tf_1_centered.params, tuple(self.mosaic_size[::-1])
        )

        return mask_image_1

    def get_image_data(
        self, image_idx: int
    ) -> Tuple[np.ndarray, AffineTransform, AffineTransform]:

        if image_idx < 0:
            raise IndexError("list index out of range")
        data = (
            self.images_list[image_idx],
            self.translations_list[image_idx],
            self.transformations_list[image_idx],
        )
        return data

    def image_intersection(
        self, idx_1: int, idx_2: int, only_translation: bool = True
    ) -> np.ndarray:
        mask_image_1 = self.mask(idx_1, only_translation)
        mask_image_2 = self.mask(idx_2, only_translation)

        return np.logical_and(mask_image_1 > 0, mask_image_2 > 0)

    def combinations(self) -> List[Tuple[int, int]]:
        indices = np.arange(len(self.images_list))
        return itertools.combinations(indices, 2)  # type: ignore

    def images_relationships(
        self, min_px_intersection=1, only_translation=True
    ) -> List[Tuple[int, int]]:
        relationships_list = []

        for idx1, idx2 in self.combinations():
            intersection = self.image_intersection(idx1, idx2, only_translation)
            if np.count_nonzero(intersection) >= min_px_intersection:
                relationships_list.append((idx1, idx2))

        return relationships_list

    def set_transforms_list(self, transforms_list: List[AffineTransform]) -> None:
        """Aplica la lista de transformaciones manteniendo el origen de coordenadas
        constante. Al finalizar, recalcula el tamaño del mosaico."""

        assert len(transforms_list) == self.n_images()
        assert all([isinstance(x, AffineTransform) for x in transforms_list])

        self.transformations_list = transforms_list
        self.update_size()

    def set_transformation(self, image_idx: int, tranformation: AffineTransform) -> None:
        self.transformations_list[image_idx] = tranformation
        self.update_size()

    def update_size(self):
        if self.n_images() == 0:
            return

        corners_list = []
        for idx in range(self.n_images()):
            image, translation, transformation = self.get_image_data(idx)
            centered_tf = self._get_centered_transform(
                translation, transformation, image.shape[:2]
            )

            corners = self._tf_corners(image, centered_tf)
            corners_list.append(corners)

        corners_list = np.array(corners_list).reshape(-1, 2)  # List of points (x, y)

        left, top = np.min(corners_list, axis=0)
        right, bottom = np.max(corners_list, axis=0)

        new_mosaic_size = (bottom - top, right - left)
        self.mosaic_size = self._round_mosaic_size(new_mosaic_size)
        self._apply_displacement_to_images(-left, -top)

    def _apply_displacement_to_images(self, dx: int, dy: int) -> None:
        translations_updated = []
        shift = np.zeros((3, 3))
        shift[:2, -1] = [dx, dy]
        for tf in self.translations_list:
            tf_updated = AffineTransform(tf.params + shift)
            translations_updated.append(tf_updated)
        self.translations_list = translations_updated

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to a dictionary format with primitive types (lists, tuples,
        floats...). The dictionary representation contains the following keys:
            - "images": A list of the images in the mosaic.
            - "translations": A list of translations (affine matrix).
            - "transformations": A list of transformation matrices (affine matrix).
            - "size": A tuple indicating the size of the mosaic.

        Returns:
            Dict[str, Any]: A dictionary representation
        """
        return {
            "images": [image.tolist() for image in self.images_list],
            "translations": [t.params.tolist() for t in self.translations_list],
            "transformations": [t.params.tolist() for t in self.transformations_list],
            "size": tuple(self.mosaic_size),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mosaic":
        """
        Create a `Mosaico` object from a dictionary with primitive types. The dictionary
        representation must contains the following keys:
            - "images": A list of the images in the mosaic.
            - "translations": A list of translations (affine matrix).
            - "transformations": A list of transformation matrices (affine matrix).
            - "size": A tuple indicating the size of the mosaic.

        Args:
            data (Dict[str, Any]): A dictionary representation

        Returns:
            Mosaico: An instance of the `Mosaico` class.
        """
        instance = Mosaic()
        instance.mosaic_size = tuple(data["size"])
        instance.images_list = [np.array(image, "uint8") for image in data["images"]]
        instance.translations_list = [
            AffineTransform(np.array(matrix)) for matrix in data["translations"]
        ]
        instance.transformations_list = [
            AffineTransform(np.array(matrix)) for matrix in data["transformations"]
        ]

        return instance


def to_centered_transform(transform, image_size):
    h, w = image_size
    tx, ty = [(w / 2), (h / 2)]

    return (
        AffineTransform(translation=(tx, ty)).params
        @ transform
        @ AffineTransform(translation=(-tx, -ty)).params
    )


def apply_transform(image, matrix, dsize, point_of_rotation=None):
    if point_of_rotation is not None:
        x, y = point_of_rotation
        matrix = (
            AffineTransform(translation=(x, y)).params
            @ matrix
            @ AffineTransform(translation=(-x, -y)).params
        )

    return cv2.warpPerspective(image, matrix, dsize)


def apply_transform_from_center(image, matrix, dsize):
    h, w = image.shape[:2]
    point_of_rotation = [(w / 2), (h / 2)]

    return apply_transform(image, matrix, dsize, point_of_rotation)
