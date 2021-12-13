import argparse
import json
import logging

import pyflann
import scipy.io as sio

from util.iou_util import IouUtil
from util.projective_camera import ProjectiveCamera
from util.synthetic_util import SyntheticUtil

LOGGER = logging.getLogger(__name__)


def retrieve_homography(retrieved_camera_data):
    u, v, fl = retrieved_camera_data[0:3]
    rod_rot = retrieved_camera_data[3:6]
    cc = retrieved_camera_data[6:9]

    retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)
    h = IouUtil.template_to_image_homography_uot(retrieved_camera)
    return h


def refine_homography(query_image, retrieved_image):
    dist_threshold = 50
    query_dist = SyntheticUtil.distance_transform(query_image)
    retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)

    query_dist[query_dist > dist_threshold] = dist_threshold
    retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

    h = SyntheticUtil.find_transform(retrieved_dist, query_dist)
    return h


def main():
    database_data = sio.loadmat(args.database)
    database_features = database_data['features']
    LOGGER.info(f'loaded database features: {database_features.shape}')

    test_data = sio.loadmat(args.test)
    test_features = test_data['features']
    LOGGER.info(f'loaded test features: {test_features.shape}')

    model_data = sio.loadmat(args.model)

    flann = pyflann.FLANN()
    flann.build_index(database_features, trees=8, checks=1000)
    result, _ = flann.nn_index(test_features, num_neighbors=1)
    LOGGER.info(f'calculated nearest neighbor database images')

    out_data = []
    for index in range(len(test_features)):
        image_id = test_data['image_ids'][index]
        query_image = test_data['edge_map'][:, :, :, index]

        retrieved_camera = database_data['cameras'][result[index]]
        retrieved_image = SyntheticUtil.camera_to_edge_image(
            retrieved_camera,
            model_data['points'], model_data['line_segment_index'],
            im_h=720, im_w=1280, line_width=4
        )

        database_h = retrieve_homography(retrieved_camera)
        refine_h = refine_homography(query_image, retrieved_image)
        h = refine_h @ database_h

        out_data.append({
            'image_id': image_id,
            'h': h.tolist(),
            'h_db': database_h.tolist(),
            'h_ref': refine_h.tolist(),
            'template_size': [74, 115],
            'image_size': [720, 1280]
        })

    with open(args.out, 'w') as f:
        json.dump(out_data, f)
    LOGGER.info(f'saved {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='テストファイル(.mat)', required=True)
    parser.add_argument('--out', help='出力ファイル(.json)', required=True)
    parser.add_argument('--database', help='データベースファイル(.mat)',
                        default='../data/features/database_camera_feature_HoG.mat')
    parser.add_argument('--model', help='モデルファイル（.mat）',
                        default='../data/worldcup2014.mat')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    main()
