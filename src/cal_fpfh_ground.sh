input_file_dir="../data/"
output_file_dir="${input_file_dir}output/fpfh/"
## generate fpfh feature and convert binary pcd to ascii
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  pcl_fpfh_estimation "${input_file_dir}Vaihingen3D_Traininig.pcd" "${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.pcd" -radius $i
  pcl_convert_pcd_ascii_binary "${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.pcd" "${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.txt" 0
  pcl_fpfh_estimation "${input_file_dir}Vaihingen3D_EVAL_WITH_REF.pcd" "${output_file_dir}Vaihingen3D_EVAL_WITH_REF_fpfh_$i.pcd" -radius $i
  pcl_convert_pcd_ascii_binary "${output_file_dir}Vaihingen3D_EVAL_WITH_REF_fpfh_$i.pcd" "${output_file_dir}Vaihingen3D_EVAL_WITH_REF_fpfh_$i.txt" 0
done


## visualizer fpfh feature
# python feature_visualization.py Vaihingen3D_Training_asZ_fpfh3.txt ./fpfh_3 -f fpfh
# python feature_visualization.py Vaihingen3D_Training_asZ_fpfh2.txt ./fpfh_2 -f fpfh
# python feature_visualization.py Vaihingen3D_Training_asZ_fpfh1.txt ./fpfh_1 -f fpfh

