import itertools
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import cv2
import numpy as np
from skimage.transform import AffineTransform


class Mosaic:
    """A class representing a mosaic of OCTA images with affine transformations."""

    def __init__(self):
        """
        Initializes the Mosaic with empty lists for images, translations,
        transformations, and an initial mosaic size of (0, 0).
        """
        self.images_list = []
        self.translations_list = []
        self.transformations_list = []

        self.mosaic_size = (0, 0)

    def __repr__(self):
        class_name = __class__.__name__
        size = self.mosaic_size
        n_images = self.n_images()
        return f"{class_name}(size={size}, n_images={n_images})"

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
        new_mosaic = Mosaic()

        new_mosaic.images_list = deepcopy(self.images_list)
        new_mosaic.translations_list = deepcopy(self.translations_list)
        new_mosaic.transformations_list = deepcopy(self.transformations_list)
        new_mosaic.mosaic_size = self.mosaic_size

        return new_mosaic

    def add(
        self,
        image: np.ndarray,
        translation: Optional[Tuple[float, float]] = None,
        tr_format: Literal["ij", "xy"] = "ij",
    ) -> None:
        """
        Add an image to the mosaic with an optional translation and transformation.

        Args:
            image (np.ndarray): The image to add.
            translation (Tuple[float, float], optional): The translation to apply (default is [0, 0]).
            tr_format (str, optional): The format of the translation ('ij' or 'xy').
        """
        if translation is None:
            translation = [0, 0]
        if tr_format == "ij":
            translation = translation[::-1]

        assert len(translation) == 2 and tr_format in ["ij", "xy"]

        tf = AffineTransform(translation=translation)
        self._update_and_add(image, tf)

    def n_images(self):
        """
        Return the number of images in the mosaic.

        Returns:
            int: The number of images in the mosaic.
        """
        return len(self.images_list)

    def _tf_point(self, xy_point: Iterable[float], tf: AffineTransform) -> np.ndarray:
        """
        Transforms a point using an affine transformation.

        Args:
            xy_point (Iterable): The point (x, y) to transform.
            tf (AffineTransform): The transformation to apply.

        Returns:
            np.ndarray: The transformed point.
        """
        point = np.array([*xy_point, 1], "float32")
        point_tf = tf.params @ point
        return point_tf[:2]

    def _tf_corners(
        self, image: np.ndarray, tf: AffineTransform
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms the corners of an image using an affine transformation.

        Args:
            image (np.ndarray): The image whose corners to transform.
            tf (AffineTransform): The transformation to apply.

        Returns:
            tuple: The transformed corners of the image.
        """
        h, w = image.shape[:2]

        top_left = self._tf_point([0, 0], tf)
        top_right = self._tf_point([w, 0], tf)
        bottom_left = self._tf_point([0, h], tf)
        bottom_right = self._tf_point([w, h], tf)

        return top_left, top_right, bottom_left, bottom_right

    def _get_centered_transform(
        self,
        translation: AffineTransform,
        transformation: AffineTransform,
        image_shape: Tuple[int, int],
    ):
        """
        Combines translation and transformation matrices and applies them to center
        the image.

        Args:
            translation (AffineTransform): The translation to apply.
            transformation (AffineTransform): The transformation to apply.
            image_shape (tuple): The shape (H x W) of the image to be transformed.

        Returns:
            AffineTransform: The combined centered transformation.
        """
        combinated = translation.params @ transformation.params
        return AffineTransform(to_centered_transform(combinated, image_shape))

    def image_centered_transform(self, idx: int) -> AffineTransform:
        """
        Returns the centered transformation for a specific image in the mosaic.

        Args:
            idx (int): The index of the image.

        Returns:
            AffineTransform: The centered transformation for the image.
        """
        translation = self.translations_list[idx]
        transformation = self.transformations_list[idx]
        image_shape = self.images_list[idx].shape[:2]
        return self._get_centered_transform(translation, transformation, image_shape)

    def _round_mosaic_size(self, size: Tuple[float, float]) -> Tuple[int, int]:
        """
        Rounds the mosaic size to the nearest integer.

        Args:
            size (tuple): The size to round.

        Returns:
            tuple: The rounded size.
        """
        return tuple(np.ceil(size).astype(int))

    def _update_and_add(
        self, new_image: np.ndarray, new_image_tf: AffineTransform
    ) -> None:
        """
        Updates the mosaic size and adds a new image with its transformation to the
        mosaic.

        Args:
            new_image (np.ndarray): The image to add.
            new_image_tf (AffineTransform): The transformation to apply.
        """
        h, w = self.mosaic_size
        new_w, new_h = w, h
        corners = self._tf_corners(new_image, new_image_tf)

        # 1. Is the new image coming out on the left?
        if np.min(corners, axis=0)[0] < 0:
            tmp = np.min(corners, axis=0)[0]
            x_shift = abs(tmp)
            new_w += abs(tmp)
        else:
            x_shift = 0

        # 2. Is the new image coming out on the right?
        if np.max(corners, axis=0)[0] > w:
            new_w += np.max(corners, axis=0)[0] - w

        # 3. Is the new image coming out on the top?
        if np.min(corners, axis=0)[1] < 0:
            tmp = np.min(corners, axis=0)[1]
            y_shift = abs(tmp)
            new_h += abs(tmp)
        else:
            y_shift = 0

        # 4. Is the new image coming out on the bottom?
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

    def image(self, pad: int = 0, rgb: bool = False) -> Optional[np.ndarray]:
        """
        Generates a mosaic image from the images in the mosaic, optionally padded.

        Args:
            pad (int, optional): Padding to apply to the images (default is 0).
            rgb (bool, optional): Whether to convert images to RGB (default is False).

        Returns:
            np.ndarray or None: The generated mosaic image or None if no images.
        """
        images_order = np.arange(len(self.images_list))[::-1]
        return self.image_with_order(images_order, pad, rgb)

    def image_with_order(
        self, images_order: Iterable[int], pad: int = 0, rgb: bool = False
    ) -> Optional[np.ndarray]:
        """
        Generates a mosaic image drawing the images based on the specified order
        (foreground to background).

        Args:
            images_order (list): A list of indices indicating the order of images.
            pad (int, optional): Padding to apply to the images (default is 0).
            rgb (bool, optional): Whether to convert images to RGB (default is False).

        Returns:
            np.ndarray or None: The generated mosaic image or None if no images.
        """
        if len(self.images_list) == 0:
            return None

        if len(images_order) == 0:
            raise ValueError("images order can not be empty.")

        shift = np.zeros((3, 3))
        shift[:2, -1] = [pad, pad]
        h, w = self.mosaic_size

        mosaic = np.pad(np.zeros((h + pad * 2, w + pad * 2)), pad)
        if rgb:
            mosaic = np.dstack((mosaic, mosaic, mosaic))

        mosaic_size = mosaic.shape[:2][::-1]
        for idx in images_order:
            image, translation, transformation = self.get_image_data(idx)
            translation_padded = AffineTransform(translation.params + shift)

            centered_tf = self._get_centered_transform(
                translation_padded, transformation, image.shape[:2]
            )
            indices = cv2.warpPerspective(
                np.ones_like(image), centered_tf.params, mosaic_size
            )
            if rgb and image.ndim == 2:
                image = np.dstack((image, image, image))

            image_tf = cv2.warpPerspective(image, centered_tf.params, mosaic_size)
            mosaic = np.where(indices > 0, image_tf, mosaic)

        return mosaic

    def mask(self, idx: int, only_translation: bool = True) -> np.ndarray:
        """
        Generates a mask for a specified image in the mosaic.

        Args:
            idx (int): The index of the image.
            only_translation (bool, optional): Whether to apply only translation (default
                is True).

        Returns:
            np.ndarray: The generated mask.
        """
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
        """
        Retrieves the image data (image, translation, and transformation) for a
        specified index.

        Args:
            image_idx (int): The index of the image.

        Returns:
            tuple: A tuple containing the image, translation, and transformation.
        """
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
        """
        Calculates the intersection of two images in the mosaic.

        Args:
            idx_1 (int): The index of the first image.
            idx_2 (int): The index of the second image.
            only_translation (bool, optional): Whether to consider only translation
                (default is True).

        Returns:
            np.ndarray: The intersection mask of the two images.
        """
        mask_image_1 = self.mask(idx_1, only_translation)
        mask_image_2 = self.mask(idx_2, only_translation)

        return np.logical_and(mask_image_1 > 0, mask_image_2 > 0)

    def combinations(self) -> List[Tuple[int, int]]:
        """
        Generates all possible combinations of two images in the mosaic.

        Returns:
            list: A list of tuples containing pairs of image indices.
        """
        indices = np.arange(len(self.images_list))
        return itertools.combinations(indices, 2)  # type: ignore

    def images_relationships(
        self, min_px_intersection=1, only_translation=True
    ) -> List[Tuple[int, int]]:
        """
        Calculates the relationships between images based on their intersection. Two
        images are related if they overlap by at least `min_px_intersection` pixels.

        Args:
            min_px_intersection (int, optional): The minimum number of intersecting pixels
                required for a relationship (default is 1).
            only_translation (bool, optional): Whether to consider only translation
                (default is True).

        Returns:
            list: A list of tuples containing pairs of image indices with enough intersection.
        """
        relationships_list = []

        for idx1, idx2 in self.combinations():
            intersection = self.image_intersection(idx1, idx2, only_translation)
            if np.count_nonzero(intersection) >= min_px_intersection:
                relationships_list.append((idx1, idx2))

        return relationships_list

    def set_transforms_list(self, transforms_list: List[AffineTransform]) -> None:
        """
        Applies a list of transformations to the mosaic, keeping the coordinate origin
        constant.

        Args:
            transforms_list (list): A list of AffineTransform objects.
        """

        assert len(transforms_list) == self.n_images()
        assert all([isinstance(x, AffineTransform) for x in transforms_list])

        self.transformations_list = transforms_list
        self.update_size()

    def set_transformation(self, image_idx: int, tranformation: AffineTransform) -> None:
        """
        Sets a specific transformation for an image in the mosaic.

        Args:
            image_idx (int): The index of the image.
            tranformation (AffineTransform): The transformation to apply.
        """
        self.transformations_list[image_idx] = tranformation
        self.update_size()

    def update_size(self) -> None:
        """
        Updates the mosaic size based on the images, translations, and transformations.
        """
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
        """
        Applies displacement to the images in the mosaic.

        Args:
            dx (int): The displacement in the x direction.
            dy (int): The displacement in the y direction.
        """
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
        Create a `mosaic` object from a dictionary with primitive types. The dictionary
        representation must contains the following keys:
            - "images": A list of the images in the mosaic.
            - "translations": A list of translations (affine matrix).
            - "transformations": A list of transformation matrices (affine matrix).
            - "size": A tuple indicating the size of the mosaic.

        Args:
            data (Dict[str, Any]): A dictionary representation

        Returns:
            mosaic: An instance of the `mosaic` class.
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
