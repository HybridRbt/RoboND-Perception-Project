#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')

    # models for perception Exercise-3
    models_ex3 = [\
       'beer',
       'bowl',
       'create',
       'disk_part',
       'hammer',
       'plastic_cup',
       'soda_can']
    # 50 loops

    # models for perception project, pick list 1
    models_1 = [\
       'biscuits',
       'soap',
       'soap2']
#   50 loops are good enough to get a ~90% result, all objects recognized

    # models for perception project, pick list 2
    models_2 = [\
       'biscuits',
       'soap',
       'book',
       'soap2',
       'glue']
#   100 loops, ~90% confusion matrix, recognized 3 ~ 4 objects
#   200 loops, ~90% confusion matrix  rbf
#   500 loops, ~0.96 accuracy rbf
#   500 loops, linear, 0.94
#   100 loops, linear, normal range 256, lowest 0.93, glue is not found
#   200 loops, linear, normal range 256, all objects found

    # models for perception project, pick list 3
    models_3 = [\
        'sticky_notes',
        'book',
        'snacks',
        'biscuits',
        'eraser',
        'soap2',
        'soap',
        'glue']
#   300 loops, ~90% confusion matrix, yet only recognize books
#   300 loops, linear, normal 256. only glue is not found

    # models = models_1
    # models = models_2
    models = models_3
    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(300):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()


    pickle.dump(labeled_features, open('training_set.sav', 'wb'))
