> cd $GIT_ROOT

> notes_to_pdf.py --input data605/lectures_md/final_enhanced_markdown_lecture_2.txt --output tmp.pdf --type slides --skip_action cleanup_after --debug_on_error --toc_type navigation --filter_by_slides 1:4

> gen_data605.sh 01

> gen_msml610.sh 02


> FILE=msml610/lectures_source/Lesson05*
> process_slides.py --in_file $FILE --action slide_format_figures --out_file $FILE --use_llm_transform
> process_slides.py --in_file $FILE --action slide_check --out_file ${FILE}.check --use_llm_transform --limit None:10

rsync -avz -e "ssh -i ~/.ssh/ck/saggese-cryptomatic.pem" saggese@$DEV1:/data/saggese/src/umd_classes1/msml610/lectures/ msml610/lectures/; open msml610/lectures/*07.1* -a "skim"
