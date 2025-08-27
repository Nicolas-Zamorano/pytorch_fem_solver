import triangle
from tensordict import TensorDict

vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]

segments = [[0, 1], [1, 2], [2, 3], [3, 0]]


def mesh_to_tensordict(mesh_dict):

    key_map = {
        "vertices": ("vertices", "coordinates"),
        "vertex_markers": ("vertices", "markers"),
        "triangles": ("triangles", "indices"),
        "neighbors": ("triangles", "neighbors"),
        "edges": ("edges", "indices"),
        "edge_markers": ("edges", "markers"),
    }

    sub_dictionaries = {
        "coords": {},
        "triangles": {},
        "edges": {},
    }

    for key, value in mesh_dict.items():
        subname, new_key = key_map[key]
        sub_dictionaries[subname][new_key] = value

    td = TensorDict(
        {
            name: (
                TensorDict(content, batch_size=[len(next(iter(content.values())))])
                if content
                else TensorDict({}, batch_size=[0])
            )
            for name, content in sub_dictionaries.items()
        },
        batch_size=[],
    )
    return td


triangulation = triangle.triangulate(
    {"vertices": vertices, "segments": segments}, "na0.005"
)

print(triangulation)

xd = mesh_to_tensordict(triangulation)
print(xd)
