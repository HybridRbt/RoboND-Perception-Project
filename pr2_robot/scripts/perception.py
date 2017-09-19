#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

from perception_helper import *

def clustering(pcl_data):
    # Create k-d tree
    white_cloud = XYZRGB_to_XYZ(pcl_data)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold
    # 0.001: too small, the cluster will be too fractrized, colored like rainbow
    # 0.01: less colorful, but still mixing multiple colors onto one object
    # 0.1: nothing will show up except for the bowl at the center
    # this means the min and max numbers need to be tweaked, more specifically,
    # increase maximum
    # when it was increased to 5000, all the objects are shown, now can get back
    # to tweak this number between 0.01 and 0.1
    # a unique color is achived for each object when this number = 0.014
    ec.set_ClusterTolerance(0.014)

    # Set minimum cluster size
    # the lower this number, the more points each cluster has
    # but it cannot be too small otherwise there will be "noises" inside each
    # object
    # ec.set_MinClusterSize(10)  # for test world 2
    ec.set_MinClusterSize(5)  # for test world 3

    # Set maximum cluster size
    # the larger this number, the more ponints will show up
    # but it cannot be too large otherwise two objects may be colored into one
    ec.set_MaxClusterSize(10000)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    return white_cloud, tree, cluster_indices

def color_clusters(pcl_data_xyz, cluster_indices):
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    # create a list for the colored cluster points
    color_cluster_point_list = []

    # traverse the indices and append to the list
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([pcl_data_xyz[indice][0],
                                             pcl_data_xyz[indice][1],
                                             pcl_data_xyz[indice][2],
                                             rgb_to_float(cluster_color[j])])
    # Create new cloud containing all clusters colored
    cluster_cloud_colored = pcl.PointCloud_PointXYZRGB()
    cluster_cloud_colored.from_list(color_cluster_point_list)
    return cluster_cloud_colored

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    pcl_original_pub.publish(pcl_msg)

    # Statistical Outlier Filtering
    pcl_noise_filtered = statistical_outlier_fiter(pcl_data)
    pcl_no_noise_pub.publish(pcl_to_ros(pcl_noise_filtered))

    # Voxel Grid Downsampling
    pcl_downsampled = voxel_downsampling(pcl_noise_filtered)
    pcl_voxel_downsampled_pub.publish(pcl_to_ros(pcl_downsampled))

    # PassThrough Filter
    pcl_passed = passthrough_filtering(pcl_downsampled)
    pcl_passthrough_pub.publish(pcl_to_ros(pcl_passed))

    # RANSAC Plane Segmentation
    inliers = plane_fitting(pcl_passed)

    # Extract inliers and outliers
    # inliers: table
    # outliers: objects
    cloud_table = extract_inliers(inliers, pcl_passed)
    cloud_objects = extract_outliers(inliers, pcl_passed)

    # Euclidean Clustering
    white_cloud, tree, cluster_indices = clustering(cloud_objects)

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud_colored = color_clusters(white_cloud, cluster_indices)

    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cluster_cloud_colored = pcl_to_ros(cluster_cloud_colored)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud_colored)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)

        # Compute the associated feature vector
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    test_scene_num = Int32()
    # test_scene_num.data = 1  # test world 1
    # test_scene_num.data = 2  # test world 2
    test_scene_num.data = 3  # test world 3

    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    # create a dictionary to hold the names and its cloud data in the object_list
    centroids_dic = {}

    # traverse the object_list, build the dictionary
    for detected_object in object_list:
        label = detected_object.label
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        centroids_dic[label] = centroid

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    drop_pose_param = rospy.get_param('/dropbox')

    for i in range(0, len(drop_pose_param)):
        if drop_pose_param[i]['name'] == 'left':
            drop_pose_left = drop_pose_param[i]['position']
        else: # right
            drop_pose_right = drop_pose_param[i]['position']

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # create the dict list
    dict_list = []
    print("num of obj: " + str(len(object_list_param)))
    # Loop through the pick list
    for i in range(0, len(object_list_param)):
        # Parse parameters into individual variables
        object_name.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        # Assign the arm to be used for pick_place
        if object_group == 'green':
            arm_name.data = 'right'
            object_drop_pos = drop_pose_right
        else:  # group == red
            arm_name.data = 'left'
            object_drop_pos = drop_pose_left

        # Get the PointCloud for a given object and obtain it's centroid
        # get the centroid from the dictionary
        centr = centroids_dic[object_name.data]
        pick_pose.position.x = np.asscalar(centr[0])
        pick_pose.position.y = np.asscalar(centr[1])
        pick_pose.position.z = np.asscalar(centr[2])

        # Create 'place_pose' for the object
        place_pose.position.x = object_drop_pos[0]
        place_pose.position.y = object_drop_pos[1]
        place_pose.position.z = object_drop_pos[2]

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')

        #try:
        #    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
        #    TEST_SCENE_NUM = test_scene_num
        #    OBJECT_NAME = object_name
        #    WHICH_ARM = arm_name
        #    PICK_POSE = pick_pose
        #    PLACE_POSE = place_pose

        #    resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

        #    print ("Response: ",resp.success)

        #except rospy.ServiceException, e:
        #    print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    print "now ouput to yaml"
    yaml_filename = 'output_1.yaml'
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('perception', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model_1.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
