if [[ 0 == 1 ]]; then
class_project/create_PR.py \
      --input_file class_project/fall2025_msml610_branches.txt \
      --source_dir /Users/saggese/src/umd_classes2 \
      --dst_dir /Users/saggese/src/umd_classes3 \
      --output_file output.txt
  else
class_project/create_PR.py \
  --input_file class_project/fall2025_msml610_branches_dirs2.txt \
  --source_dir /Users/saggese/src/umd_classes2 \
  --dst_dir /Users/saggese/src/umd_classes3 \
  --copy_dirs
fi;
