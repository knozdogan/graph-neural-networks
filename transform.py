import torch
import numpy as np
from torch_scatter import scatter_mean

from torch_geometric.data import Data

def cxcywh_to_xyxy(arr: torch.Tensor) -> torch.Tensor:
    """arr: m, 4"""

    m = arr.size(0)
    xyxy = torch.zeros(m,4)
    xyxy[:,:2] = arr[:,:2] - arr[:,2:]/2
    xyxy[:,2:] = arr[:,:2] + arr[:,2:]/2

    return xyxy

class ToRAG(object):

    def __init__(self, add_bboxes_node_attr: bool = True, add_seg: bool = True, add_img: bool = True, iou_threshold=0.1, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.add_bboxes_node_attr = add_bboxes_node_attr
        self.iou_threshold = iou_threshold
        self.kwargs = kwargs

    def __call__(self, gpr) -> Data:
        from skimage.segmentation import slic
        from skimage.future import graph
        from skimage import filters
        from torch_geometric.utils.convert import from_networkx
        from networkx import clustering
        from skimage.measure import regionprops


        # numpy array (h, w)
        img = gpr['img']
        h, w = img.shape

        segments = slic(
            img, compactness=self.kwargs["compactness"], 
            n_segments=self.kwargs["n_segments"], 
            sigma=self.kwargs["sigma"], start_label=0
        )

        regions = regionprops(segments+1)
        edges = filters.sobel(img)
        G = graph.rag_boundary(segments, edges)
        data = from_networkx(G, group_edge_attrs=['weight'])

        img = torch.from_numpy(img[...,np.newaxis])
        h, w, c = img.size()

        seg = torch.from_numpy(segments)
        x = scatter_mean(img.view(h * w, c), seg.view(h * w), dim=0) # sorted node labels => 0,1,2,3 ...

        clustering_coeff = clustering(G, weight='weight')
        degrees = G.degree(weight='weight')
        new_x = torch.zeros(x.size(0), 3) # pixel_mean_val, clustering_coeff, degree
        new_x[:,0] = x[:,0]

        data.pos = torch.zeros(x.size(0), 2) # central_x_coord, central_y_coord
        
        target_bboxes = cxcywh_to_xyxy(torch.tensor(gpr['bboxes']))
        data.bboxes = target_bboxes.view(1,target_bboxes.size(0),target_bboxes.size(1))

        for region in regions:
            ch, cw = region.centroid
            data.pos[region.label-1,0] = cw/w #x
            data.pos[region.label-1,1] = ch/h #y
            new_x[region.label-1,2] = clustering_coeff[region.label-1]
            new_x[region.label-1,1] = degrees[region.label-1]

        data.regions = regions
        data.x = new_x
        data.y = torch.zeros(x.size(0),1)

        for t in target_bboxes:
            uniq_values = np.unique(
                segments[int(t[1]*h):int(t[3]*h),int(t[0]*w):int(t[2]*w)]
            )
            data.y[uniq_values,:] = 1
        
        if self.add_seg:
            data.seg = seg.view(1, h, w)

        if self.add_img:
            data.img = img.permute(2, 0, 1).view(1, c, h, w)

        return data
