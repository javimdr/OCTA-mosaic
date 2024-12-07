import itertools
import warnings

import numpy as np

from octa_mosaic.modules.mosaic import Mosaic
from octa_mosaic.modules.template_matching import template_matching


def get_corner(center, image_shape, corner="top_left"):
    H, W = image_shape
    top_left = np.subtract(center, [H // 2, W // 2])

    if corner == "top_left":
        return top_left
    elif corner == "top_right":
        return np.add(top_left, [0, W])
    elif corner == "bottom_left":
        return np.add(top_left, [H, 0])
    elif corner == "bottom_right":
        return np.add(top_left, [H, W])
    else:
        raise ValueError("Unknown corner.")


class Mosaico_TM_register:
    def cc_order(self, images_list):
        warnings.warn(
            "El orden del primer par es 'arbitrario'. Mejor usar 'cc_order_new()'."
        )
        images_order = []
        images_locations = []

        images_dict = {idx: image for idx, image in enumerate(images_list)}
        index_A, index_B, loc_B = self._select_first_pair(images_dict)

        mosaico = Mosaic()
        mosaico.add(images_dict[index_A])
        mosaico.add(images_dict[index_B], loc_B)

        images_order = [index_A, index_B]
        images_locations = [(0, 0), tuple(loc_B)]

        del images_dict[index_A]
        del images_dict[index_B]

        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaico, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaico.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        #         print(out_str)
        return images_order, images_locations

    def cc_order_new(self, images_list):
        images_order = []
        images_locations = []

        images_dict = {idx: image for idx, image in enumerate(images_list)}
        index_A, index_B, loc_B = self._select_first_pair(images_dict)

        mosaico_AB = Mosaic()
        mosaico_AB.add(images_dict[index_A])
        mosaico_AB.add(images_dict[index_B], loc_B)

        mosaico_BA = Mosaic()
        mosaico_BA.add(images_dict[index_B])
        mosaico_BA.add(images_dict[index_A], -loc_B)

        #          plots.plot_mult([mosaico_AB.image(), mosaico_BA.image()], cols=2)

        del images_dict[index_A]
        del images_dict[index_B]

        image_idx_AB, location_AB, cc_AB = self._select_next_image(
            mosaico_AB, images_dict
        )
        image_idx_BA, location_BA, cc_BA = self._select_next_image(
            mosaico_BA, images_dict
        )

        mosaico_AB.add(images_dict[image_idx_AB], location_AB)
        mosaico_BA.add(images_dict[image_idx_BA], location_BA)

        # plots.plot_mult([mosaico_AB.image(), mosaico_BA.image()], [f"{cc_AB:.4f}, ({cc_AB > cc_BA})", f"{cc_BA:.4f}, ({cc_BA > cc_AB})"], cols=2, base_size=10)
        out_str = ""
        if cc_AB > cc_BA:
            out_str += "Mosaico (AB)."
            images_order = [index_A, index_B, image_idx_AB]
            images_locations = [(0, 0), tuple(loc_B), tuple(location_AB)]

            del images_dict[image_idx_AB]
            mosaico = mosaico_AB

        else:
            out_str += "Mosaico (BA)."
            images_order = [index_B, index_A, image_idx_BA]
            images_locations = [(0, 0), tuple(-loc_B), tuple(location_BA)]

            del images_dict[image_idx_BA]
            mosaico = mosaico_BA

        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaico, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaico.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        #         print(out_str)
        return images_order, images_locations

    def _select_rest(self, images_dict, mode="full", corr_func="ZNCC"):
        pass

    def _select_first_pair(self, images_dict, mode="full", corr_func="ZNCC"):
        loop_cc = []
        loop_loc_corners = []

        images_idx_pairs = list(itertools.combinations(list(images_dict.keys()), 2))
        for index_A, index_B in images_idx_pairs:
            image_A = images_dict[index_A]
            image_B = images_dict[index_B]

            cc_value, cc_loc, cc_matrix = template_matching(
                image_A, image_B, mode=mode, corr_func=corr_func
            )
            cc_loc = cc_loc[::-1]
            if mode == "full":
                cc_loc = get_corner(cc_loc, image_B.shape[:2])

            cc_loc = get_corner(cc_loc, image_B.shape[:2])
            loop_cc.append(cc_value)
            loop_loc_corners.append(cc_loc)

        best_case_idx = np.argmax(loop_cc)

        index_A = images_idx_pairs[best_case_idx][0]
        index_B = images_idx_pairs[best_case_idx][1]
        IB_center = loop_loc_corners[best_case_idx]

        return index_A, index_B, IB_center

    def _select_next_image(self, mosaic, images_dict, mode="full", corr_func="ZNCC"):
        loop_cc = []
        loop_loc_corners = []
        images_dict_keys = list(images_dict.keys())
        for idx_image in images_dict_keys:
            image = images_dict[idx_image]
            cc_value, cc_loc, cc_matrix = template_matching(
                mosaic.image(), image, mode=mode, corr_func=corr_func
            )
            cc_loc = cc_loc[::-1]
            if mode == "full":
                cc_loc = get_corner(cc_loc, image.shape[:2])

            cc_loc = get_corner(cc_loc, image.shape[:2])
            loop_cc.append(cc_value)
            loop_loc_corners.append(cc_loc)

        best_case_idx = np.argmax(loop_cc)

        image_idx = images_dict_keys[best_case_idx]
        location = loop_loc_corners[best_case_idx]
        return image_idx, location, np.max(loop_cc)

    def mosaico_from_indices(self, images_order, locations, images_list):
        mosaico = Mosaic()

        for image_idx, location in zip(images_order, locations):
            mosaico.add(images_list[image_idx], location)

        return mosaico


# def f(images_order: Dict[str, Dict], filename: str) -> None:
#     for k in new_order_dict:
#         new_order_dict[k]["IMAGES_ORDER"] = [
#             int(i) for i in new_order_dict[k]["IMAGES_ORDER"]
#         ]
#         new_order_dict[k]["LOCATIONS"] = [
#             (int(t[0]), int(t[1])) for t in new_order_dict[k]["LOCATIONS"]
#         ]

#     mosaico_order_json = json.dumps(new_order_dict, indent=2)
#     with open("mosaico_order_old.json", "w") as f:
#         f.write(mosaico_order_json)


# class Mosaico_TM_register_New:
#     def __init__(self, mode: str = "full", corr_func: str = "ZNCC"):
#         self.mode = mode
#         self.corr_func = corr_func
#         if self.corr_func != corr_func:
#             warnings.warn("El comportamiento sin emplear la función 'ZNCC' no está testeado.")

#     def _tm_location_to_top_left_point(self, tm_location, template_shape):
#         # El formato de la localización devuelta por la función 'template_matching' es 'xy'
#         tm_location = tm_location[::-1]  # 'xy' to 'ij'
#         if self.mode == "full":
#             tm_location = get_corner(tm_location, template_shape)
#         tm_location = get_corner(tm_location, template_shape)

#         return Point(*tm_location[::-1])

#     def _select_first_pair(self, images_dict: Dict[int, np.ndarray]) -> Tuple[ImagePosition, ImagePosition]:
#         pairs_cc_list = []
#         pairs_list = []

#         images_idx_pairs = list(itertools.combinations(list(images_dict.keys()), 2))
#         for index_A, index_B in images_idx_pairs:
#             image_A = images_dict[index_A]
#             image_B = images_dict[index_B]

#             cc_value, location, cc_matrix = template_matching(image_A, image_B, mode=self.mode, corr_func=self.corr_func)
#             point_location = self._tm_location_to_top_left_point(location, image_B.shape[:2])

#             pair = (MosaicImage(index_A, Point(), MosaicImage(index_B, point_location))

#             pairs_cc_list.append(cc_value)
#             pairs_list.append(pair)

#         return pairs_list[np.argmax(pairs_cc_list)]


#     def _first_pair_order_based_on_edge(self,
#                                         mosaico_order: MosaicOrder,
#                                         first_pair: Tuple[ImagePosition, ImagePosition],
#                                         images_dict: Dict[int, np.ndarray],
#                                         edge_px_list: List[int],
#                                         edge_weight_list: List[int] = None
#                                        ) -> MosaicOrder

#         if edge_weight_list is None:
#             edge_weight_list = np.ones(len(edge_px_list))
#         assert len(edge_weight_list) == len(edge_px_list)

#         image_A = images_dict[first_pair[0].image_idx]
#         location_A = first_pair[0].location.as_array()  # == (0,0)
#         image_B = images_dict[first_pair[1].image_idx]
#         location_B = first_pair[1].location.as_array()


#         mosaico_AB = Mosaico()
#         mosaico_AB.add(image_A, location_A)
#         mosaico_AB.add(image_B, location_B)
#         CC_AB = optimization_utils.calc_cc_multiples_bordes(mosaico_AB, self.borders_width, self.borders_weight)

#         mosaico_BA = Mosaico()
#         mosaico_BA.add(image_B,  location_A)
#         mosaico_BA.add(image_A, -location_B)
#         CC_BA = optimization_utils.calc_cc_multiples_bordes(mosaico_BA, self.borders_width, self.borders_weight)

#         if CC_AB > CC_BA:
#             mosaico = mosaico_AB
#             images_order = [index_A, index_B]
#             images_locations = [(0,0), tuple(loc_B)]
#         else:
#             mosaico = mosaico_BA
#             images_order = [index_B, index_A]
#             images_locations = [(0,0), tuple(-loc_B)]


#     def __select_rest(self, mosaico_order, images_dict):
#         while len(images_dict) > 0:
#             image_idx, location, cc = self._select_next_image(mosaico, images_dict)
#             location = tuple(location)

#             images_order.append(image_idx)
#             images_locations.append(location)
#             mosaico.add(images_dict[image_idx], location)

#             del images_dict[image_idx]

#     def run(self, image_list: List[np.ndarray], criteria = None: str, **kwargs) -> MosaicoOrder:

#         mosaico_order = MosaicOrder()
#         images_dict = {idx: image for idx, image in enumerate(images_list)}

#         first_pair = self._select_first_pair(images_dict)

#         if criteria == "cc_of_third_image":
#             pass
#         elif criteria == "cc_of_edge":
#             mosaico_order = self._first_pair_order_based_on_edge(mosaico_order, first_pair, images_dict, **kwargs)
#         else:
#             mosaico_order = self._first_pair_order_non_criteria(mosaico_order, first_pair, images_dict)

#         for idx in mosaico_order.images_order():
#            del images_dict[idx]


#         mosaico_order = self._select_rest(mosaico_order, images_dict)
#         return mosaico_order


#         _select_order_first_pair(mosaico_order, first_pair, images_dict)
#         mosaico = Mosaico()
#         mosaico.add(images_dict[index_A])
#         mosaico.add(images_dict[index_B], loc_B)

#         images_order = [index_A, index_B]
#         images_locations = [(0,0), tuple(loc_B)]

#         del images_dict[index_A]
#         del images_dict[index_B]


# #         print(out_str)
#         return images_order, images_locations

#     def cc_order_new(self, image_list):
#         images_order = []
#         images_locations = []

#         images_dict = {idx: image for idx, image in enumerate(images_list)}
#         index_A, index_B, loc_B = self._select_first_pair(images_dict)

#         mosaico_AB = Mosaico()
#         mosaico_AB.add(images_dict[index_A])
#         mosaico_AB.add(images_dict[index_B], loc_B)

#         mosaico_BA = Mosaico()
#         mosaico_BA.add(images_dict[index_B])
#         mosaico_BA.add(images_dict[index_A], -loc_B)

# #          plots.plot_mult([mosaico_AB.image(), mosaico_BA.image()], cols=2)

#         del images_dict[index_A]
#         del images_dict[index_B]

#         image_idx_AB, location_AB, cc_AB = self._select_next_image(mosaico_AB, images_dict)
#         image_idx_BA, location_BA, cc_BA = self._select_next_image(mosaico_BA, images_dict)

#         mosaico_AB.add(images_dict[image_idx_AB], location_AB)
#         mosaico_BA.add(images_dict[image_idx_BA], location_BA)

#         #plots.plot_mult([mosaico_AB.image(), mosaico_BA.image()], [f"{cc_AB:.4f}, ({cc_AB > cc_BA})", f"{cc_BA:.4f}, ({cc_BA > cc_AB})"], cols=2, base_size=10)
#         out_str = ""
#         if cc_AB > cc_BA:
#             out_str += "Mosaico (AB)."
#             images_order = [index_A, index_B, image_idx_AB]
#             images_locations = [(0,0), tuple(loc_B), tuple(location_AB)]

#             del images_dict[image_idx_AB]
#             mosaico = mosaico_AB

#         else:
#             out_str += "Mosaico (BA)."
#             images_order = [index_B, index_A, image_idx_BA]
#             images_locations = [(0,0), tuple(-loc_B), tuple(location_BA)]

#             del images_dict[image_idx_BA]
#             mosaico = mosaico_BA

#         while(len(images_dict) > 0):
#             image_idx, location, cc = self._select_next_image(mosaico, images_dict)
#             location = tuple(location)

#             images_order.append(image_idx)
#             images_locations.append(location)
#             mosaico.add(images_dict[image_idx], location)

#             del images_dict[image_idx]

# #         print(out_str)
#         return images_order, images_locations


#     def _select_rest(self, images_dict, mode="full", corr_func="ZNCC"):
#         pass


#     def _select_next_image(self, mosaic, images_dict, mode="full", corr_func="ZNCC"):
#         loop_cc = []
#         loop_loc_corners = []
#         images_dict_keys = list(images_dict.keys())
#         for idx_image in images_dict_keys:
#             image = images_dict[idx_image]
#             cc_value, cc_loc, cc_matrix = template_matching(mosaic.image(), image, mode=mode, corr_func=corr_func)
#             cc_loc = cc_loc[::-1]
#             if mode == "full":
#                 cc_loc = get_corner(cc_loc, image.shape[:2])

#             cc_loc = get_corner(cc_loc, image.shape[:2])
#             loop_cc.append(cc_value)
#             loop_loc_corners.append(cc_loc)

#         best_case_idx = np.argmax(loop_cc)


#         image_idx = images_dict_keys[best_case_idx]
#         location = loop_loc_corners[best_case_idx]
#         return image_idx, location, np.max(loop_cc)


#     def mosaico_from_indices(self, images_order, locations, images_list):
#         mosaico = Mosaico()

#         for image_idx, location in zip(images_order, locations):
#             mosaico.add(images_list[image_idx], location)

#         return mosaico


# @dataclass
# class ImageLocation:
#     i: int = 0
#     j: int = 0

#     def __post_init__(self):
#         self.i = int(i)
#         self.j = int(j)

#     def value(self, form='ij'):
#         if form == 'ij':
#             return (self.i, self.j)
#         elif form == 'xy':
#             return (self.j, self.i)
#         else:
#             raise ValueError()

#     def as_array(self, form='ij')
#         return np.array(self.value(form))


# @dataclass
# class MosaicImage:
#     """ Represents a image index and its location in a Mosaic """
#     image_idx: int
#     location: Point

#     def __post_init__(self):
#         self.image_idx = int(image_idx)


# class MosaicOrder:
#     def __init__(self):
#         self.images_positions : List[ImagePosition] = []

#     def images_order(self) -> List[int]:
#         return [image_position.image_idx for image_position in self.images_positions]

#     def images_locations(self) -> List[Point]:
#         return [image_position.location for image_position in self.images_positions]

#     def add(self, image_position: ImagePosition) -> None:
#         self.images_positions.append(image_position)

#     def get(self, idx):
#         if idx > len(self.images_positions):
#             raise IndexError('list index out of range')

#         return self.images_positions[idx]

#     def to_json(self):
#         pass

#     def from_json(self):
#         pass

# def mosaicos_order_to_json(mosaicos_order_dict: Dict[str, MosaicoOrder], filename):
#     pass

# def mosaicos_order_from_json(filename: str) -> Dict[str, MosaicoOrder]:
#     pass
# # def save_order_to_json(mosaicos_order_dict: Dict[str, MosaicoOrder]):
# #     json_data = {}
# #     for case, mosaico_order in mosaicos_dict.items():
# #         json_data[case] = {MosaicoOrder.__dict__}

# #     for k in new_order_dict:
# #         new_order_dict[k]['IMAGES_ORDER'] = [int(i) for i in new_order_dict[k]['IMAGES_ORDER']]
# #         new_order_dict[k]['LOCATIONS'] = [(int(t[0]), int(t[1])) for t in new_order_dict[k]['LOCATIONS']]

# #     mosaico_order_json = json.dumps(new_order_dict, indent=2)
# #     with open('mosaico_order_old.json', 'w') as f:
# #         f.write(mosaico_order_json)
