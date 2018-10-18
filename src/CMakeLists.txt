cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


# project(feature)
# find_package(PCL 1.8 REQUIRED COMPONENTS common io filters sample_consensus segmentation search kdtree features surface octree registration recognition geometry keypoints ml) 
# find_package(Boost COMPONENTS filesystem system REQUIRED)
# add_executable(pcl_pfh_estimation pfh_estimation.cpp)
# target_link_libraries(pcl_pfh_estimation pcl_common pcl_io pcl_features pcl_kdtree boost_system)

project(PCL_TOOL)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pmf_tool pmf_tool.cpp)
target_link_libraries(pmf_tool ${PCL_LIBRARIES} boost_system)
