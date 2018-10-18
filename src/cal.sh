input_file_dir="../data/"
output_file_dir="${input_file_dir}output/fpfh/"

:<<EOF 
## function: generate fpfh feature and convert binary pcd to ascii

for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  pcl_fpfh_estimation "${input_file_dir}Vaihingen3D_Traininig.pcd" "${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.pcd" -radius $i
  pcl_convert_pcd_ascii_binary "${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.pcd" "${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.txt" 0
  pcl_fpfh_estimation "${input_file_dir}Vaihingen3D_EVAL_WITH_REF.pcd" "${output_file_dir}Vaihingen3D_EVAL_WITH_REF_fpfh_$i.pcd" -radius $i
  pcl_convert_pc d_ascii_binary "${output_file_dir}Vaihingen3D_EVAL_WITH_REF_fpfh_$i.pcd" "${output_file_dir}Vaihingen3D_EVAL_WITH_REF_fpfh_$i.txt" 0
done
EOF


# :<<EOF
## visualizer fpfh feature
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  input_feature_name="${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.txt"
  output_visuale_path="${output_file_dir}visualization/fpfh_$i"
  python feature_visualization.py ${input_feature_name} ${output_visuale_path} -f fpfh
done
# EOF
