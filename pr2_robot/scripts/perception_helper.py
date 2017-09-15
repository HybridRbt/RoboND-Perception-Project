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
    ec.set_MinClusterSize(5)

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

def statistical_outlier_fiter(pcl_data):
    # Start by creating a filter object:
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    pcl_filtered = outlier_filter.filter()
    return pcl_filtered

def voxel_downsampling(pcl_data):
    # Voxel Grid filtering
    # Create a VoxelGrid filter object for out input point cloud
    vox = pcl_data.make_voxel_grid_filter()

    # choose a voxel (leaf) size
    LEAF_SIZE = 0.01

    # Set voxel size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # call the filter funciton to obtain the resultant downsampled point cloud
    pcl_filtered = vox.filter()
    return pcl_filtered

def passthrough_filtering(pcl_data):
    # PassThrough filtering
    # Create a PassThrough filter objects
    passthrough = pcl_data.make_passthrough_filter()

    # Assign axis and range to the passthrough filter objects
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)

    # set the limits
    axis_min = 0.6  # this retains the table and the objects
    axis_max = 1.1

    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally, use the filter function to obtain the resultant point cloud
    pcl_filtered = passthrough.filter()
    return pcl_filtered

def plane_fitting(pcl_data):
    # RANSAC plane segmentation
    # Create the segmentation object
    seg = pcl_data.make_segmenter()

    # Set the model you wish to filter
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Note: in lesson 3-15, the quizz for this number claims it's 0.01
    # but in that case the front of the table will show, and will keep showing
    # until increased to 0.034. but in this case the bottom of the bowl will be
    # cut. Need to figure out which number to take.
    max_distance = 0.035 # this leaves only the table
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    return inliers

def extract_inliers(inliers, pcl_data):
    # Extract inliers
    extracted_inliers = pcl_data.extract(inliers, negative=False)
    return extracted_inliers

def extract_outliers(inliers, pcl_data):
    # Extract outliers
    extracted_outliers = pcl_data.extract(inliers, negative=True)
    return extracted_outliers
