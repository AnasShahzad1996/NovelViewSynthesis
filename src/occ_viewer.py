import torch
import sys
from pathlib import Path
import numpy as np

import pyvista as pv


def viz(occ_folder: Path):
    meshes = {}
    for f in occ_folder.iterdir():
        occ_padded = torch.load(f).cpu().numpy()

        bs = 256 / int(f.stem)

        grid = pv.UniformGrid()
        grid.dimensions = np.array(occ_padded.shape) + 1
        grid.spacing = (bs, bs, bs)
        grid.cell_data["values"] = occ_padded.flatten(order="F") * 255

        nu = grid.cast_to_unstructured_grid()

        nu = nu.extract_cells(nu.cell_data["values"].nonzero(), progress_bar=True)

        meshes[f.stem] = nu

    pl = pv.Plotter(line_smoothing=True)

    def callback(bs_str):
        pl.clear_actors()
        pl.add_mesh(meshes[bs_str], show_edges=True, cmap=["white"], show_scalar_bar=False, render=True)
        pl.update()

    pl.add_text_slider_widget(callback, sorted(list(meshes.keys()), key=lambda x: int(x), reverse=True), value=0)
    pl.show()



if __name__ == '__main__':
    name = sys.argv[1]

    occ_folder = Path(__file__).parent / "logs" / "paper" / "pretrain_occupancy_multi" / name

    viz(occ_folder)
