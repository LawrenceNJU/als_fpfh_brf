/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *  Copyright (c) 2014, RadiantBlue Technologies, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 */

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

// define my 
struct IsGroundPoint
{
  int isground;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IsGroundPoint, (int, isground, isground))

Eigen::Vector4f translation;
Eigen::Quaternionf orientation;

void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s input1.pcd input2.pcd output.pcd <options>\n", argv[0]);
  print_info ("  where input data are:\n");
  print_info ("                     input1.pcd is the data including fpfh features\n");
  print_info ("                     input2.pcd is the data including isground label\n");
  print_info ("                     output.pcd is the data input1.pcd + input2.pcd's isground label\n");
}

bool
loadCloud (const std::string &filename, pcl::PCLPointCloud2 &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile (filename, cloud, translation, orientation) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
concatenatePCD (const pcl::PCLPointCloud2::ConstPtr &input1, const pcl::PCLPointCloud2::ConstPtr &input2, pcl::PCLPointCloud2 &output)
{
  // Estimate
  TicToc tt;
  tt.tic ();
  pcl::PointCloud<IsGroundPoint>::Ptr input2_Ground (new pcl::PointCloud<IsGroundPoint>);
  fromPCLPointCloud2 (*input2, *input2_Ground);

  print_highlight (stderr, "concatenating ");
  
  pcl::PCLPointCloud2 output_Ground;
  toPCLPointCloud2 (*input2_Ground, output_Ground);
  concatenateFields (*input1, output_Ground, output);

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
}

void
saveCloud (const std::string &filename, const pcl::PCLPointCloud2 &output)
{
  TicToc tt; 
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());

  io::savePCDFile(filename, output, translation, orientation, true);

  print_info ( "[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", output.width * output.height); print_info (" points]\n");
}


/* ---[ */
int
main (int argc, char** argv)
{
  print_info ("Concatenate two PCD file. For more information, use: %s -h\n", argv[0]);
  if (argc < 4)
  {
    printHelp (argc, argv);
    return (-1);
  }


  // Parse the command line arguments for .pcd files
  std::vector<int> p_file_indices;
  p_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
  if (p_file_indices.size () != 3)
  {
    print_error ("Need one input PCD file and one output PCD file to continue.\n");
    return (-1);
  }
  
  // Load the first file
  pcl::PCLPointCloud2::Ptr input1 (new pcl::PCLPointCloud2);
  if (!loadCloud (argv[p_file_indices[0]], *input1))
    return (-1);

  // Load the second file
  pcl::PCLPointCloud2::Ptr input2 (new pcl::PCLPointCloud2);
  if (!loadCloud (argv[p_file_indices[1]], *input2))
    return (-1);
  
  // Perform ground filter
  pcl::PCLPointCloud2 output;
  concatenatePCD (input1, input2, output);

  // Save into the third file
  saveCloud (argv[p_file_indices[2]], output);
  return 0;
}

