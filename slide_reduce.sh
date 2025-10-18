SRC_NAME=$(ls $DIR/lectures_source/Lesson$1); echo $SRC_NAME
DST_NAME=process_slides.txt
process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $SRC_NAME --use_llm_transform --limit 0:10
vimdiff $SRC_NAME process_slides.txt
