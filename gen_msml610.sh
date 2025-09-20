#
LESSON=$1
SRC_NAME=$(cd msml610/lectures_source; ls Lesson${LESSON}*); DST_NAME=$(echo $SRC_NAME | sed 's/\.txt$/.pdf/'); notes_to_pdf.py --input msml610/lectures_source/$SRC_NAME --output msml610/lectures/$DST_NAME --type slides --toc_type navigation --debug_on_error --filter_by_slides 1:20
