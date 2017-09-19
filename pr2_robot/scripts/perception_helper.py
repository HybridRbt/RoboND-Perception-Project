# Import PCL module
import pcl

def statistical_outlier_fiter(pcl_data):
    # Start by creating a filter object:
    outlier_filter = pcl_data.make_statistical_outlier_filter()

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
    print("voxel_downsampling is done")
    #filename = 'voxel_downsampled.pcd'
    #pcl.save(pcl_filtered, filename)
    return pcl_filtered

def passthrough_filtering(filter_axis, min_limit, max_limit, pcl_data):
    # PassThrough filtering
    # Create a PassThrough filter objects
    passthrough = pcl_data.make_passthrough_filter()

    # Assign axis and range to the passthrough filter objects
    axis = filter_axis
    passthrough.set_filter_field_name(axis)

    # set the limits
    axis_min = min_limit  # this retains the table and the objects
    axis_max = max_limit

    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally, use the filter function to obtain the resultant point cloud
    pcl_filtered = passthrough.filter()

    print("pass_through is done")
    #filename = 'pass_through_filtered.pcd'
    #pcl.save(pcl_filtered, filename)
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
    print("inliers extracted")
    #filename = 'extracted_inliers.pcd'
    #pcl.save(extracted_inliers, filename)
    return extracted_inliers

def extract_outliers(inliers, pcl_data):
    # Extract outliers
    extracted_outliers = pcl_data.extract(inliers, negative=True)
    print("outliers extracted")
    #filename = 'extracted_outliers.pcd'
    #pcl.save(extracted_outliers, filename)
    return extracted_outliers
