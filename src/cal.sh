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


:<<EOF
## visualizer fpfh feature
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  input_feature_name="${output_file_dir}Vaihingen3D_Traininig_fpfh_$i.txt"
  output_visuale_path="${output_file_dir}visualization/fpfh_$i"
  python feature_visualization.py ${input_feature_name} ${output_visuale_path} -f fpfh
done
EOF

:<<EOF
## calculate ground point
for max_window_size in 30 35 40 45; do
  for slope in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do 
    for max_distance in 1 2 3 4 5 6 7 8 9 10; do
      for initial_distance in 0.05 0.1 0.15; do
	output_trainfile_name="${input_file_dir}output/train_ground/mw${max_window_size}_s${slope}_md${max_distance}_id${initial_distance}.pcd"
	./build/pmf_tool "${input_file_dir}Vaihingen3D_Traininig.pcd" ${output_trainfile_name} -max_window_size ${max_window_size} -slope ${slope} -max_distance ${max_distance} -initial_distance ${initial_distance}
	
	output_testfile_name="${input_file_dir}output/test_ground/mw${max_window_size}_s${slope}_md${max_distance}_id${initial_distance}.pcd"
	./build/pmf_tool "${input_file_dir}Vaihingen3D_EVAL_WITH_REF.pcd" ${output_testfile_name} -max_window_size ${max_window_size} -slope ${slope} -max_distance ${max_distance} -initial_distance ${initial_distance}
      don e
    done 
  done
done
EOF

## function: concatenate fpfh feature and ground and than transform pcd to ascii
train_ground_file="${input_file_dir}output/ground/train_mw30_s0.4_md1_id0.05.pcd"
test_ground_file="${input_file_dir}output/ground/test_mw30_s0.4_md1_id0.05.pcd"
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  fpfh_file="${input_file_dir}output/fpfh/Vaihingen3D_EVAL_WITH_REF_fpfh_${i}.pcd"
  fpfh_ground_file="${input_file_dir}output/fpfh_ground/Vaihingen3D_EVAL_WITH_REF_fpfh_${i}_ground.pcd"
  fpfh_ground_file_ascii="${input_file_dir}output/fpfh_ground/Vaihingen3D_EVAL_WITH_REF_fpfh_${i}_ground.txt"
  ./build/pcd_concatenate ${fpfh_file} ${test_ground_file} ${fpfh_ground_file}
  pcl_convert_pcd_ascii_binary ${fpfh_ground_file} ${fpfh_ground_file_ascii} 0
done
rm *.pcd

