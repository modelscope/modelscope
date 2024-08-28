import torch


class decoder_default:
    def __init__(self, weight=1, use_weight_map=False):
        self.weight = weight
        self.use_weight_map = use_weight_map

    def _make_grid(self, h, w):
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h - 1) * 2 - 1,
            torch.arange(w).float() / (w - 1) * 2 - 1)
        return yy, xx

    def get_coords_from_heatmap(self, heatmap):
        """
            inputs:
            - heatmap: batch x npoints x h x w

            outputs:
            - coords: batch x npoints x 2 (x,y), [-1, +1]
            - radius_sq: batch x npoints
        """
        batch, npoints, h, w = heatmap.shape
        if self.use_weight_map:
            heatmap = heatmap * self.weight

        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

        yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        coords = torch.stack([xx_coord, yy_coord], dim=-1)

        return coords
