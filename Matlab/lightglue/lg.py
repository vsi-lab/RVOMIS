import os
import sys

import numpy as np
import torch

# Prefer bundled LightGlue submodule: Matlab/lightglue/LightGlue
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMODULE_ROOT = os.path.join(_THIS_DIR, "LightGlue")
if os.path.isdir(_SUBMODULE_ROOT) and _SUBMODULE_ROOT not in sys.path:
    sys.path.insert(0, _SUBMODULE_ROOT)

from lightglue import LightGlue, SuperPoint, SIFT
from lightglue.utils import rbd, load_image


def match_features(img1_single, img2_single, device='cuda', max_keypoints=4096):

    try:
       

        extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features='sift').eval().to(device)

        #extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        #matcher = LightGlue(features='superpoint').eval().to(device)

        image0 = load_image(img1_single).cuda()
        image1 = load_image(img2_single).cuda()

        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        print(feats0.keys())

        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        matches = matches01['matches']
        scores = matches01['scores']

        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_matches = matches[sorted_indices]

        points0 = feats0['keypoints'][sorted_matches[:, 0]].cpu().numpy()
        points1 = feats1['keypoints'][sorted_matches[:, 1]].cpu().numpy()

        points0_homo = np.vstack([points0.T, np.ones((1, points0.shape[0]))])
        points1_homo = np.vstack([points1.T, np.ones((1, points1.shape[0]))])

        return points0_homo, points1_homo

    except Exception as e:
        print(f"Error: {str(e)}")
        return np.empty((3,0)), np.empty((3,0))
