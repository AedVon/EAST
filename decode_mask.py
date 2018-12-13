import numpy as np


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
            (x - 1, y), (x + 1, y), \
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >= 0 and x < w and y >= 0 and y < h;


def decode_image_by_join(score_mask):
    points = zip(*np.where(score_mask))
    h, w = np.shape(score_mask)
    group_mask = dict.fromkeys(points, -1)

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)

        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(score_mask, dtype=np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    # join by link
    for point in points:
        y, x = point
        neighbours = get_neighbours_8(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):
                pixel_cls = score_mask[ny, nx]
                if pixel_cls:
                    join(point, (ny, nx))

    mask = get_all()
    return mask

