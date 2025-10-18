# Generate the script.
> i docker_bash --base-image=623860924167.dkr.ecr.eu-north-1.amazonaws.com/cmamp --skip-pull

docker> sudo /bin/bash -c "(source /venv/bin/activate; pip install --upgrade openai)"

docker> generate_slide_script.py \
  --in_file data605/lectures_source/Lesson01-Intro.txt \
  --out_file data605/lectures_source/Lesson01-Intro.script.txt \
  --slides_per_group 3 \
  --limit 1:5

docker> gen_data605_script.sh 01 --limit 1:5

# Check correctness of all the slides.

```
SRC_NAME=$(ls $DIR/lectures_source/Lesson02*); echo $SRC_NAME
DST_NAME=process_slides.txt
docker> process_slides.py --in_file $SRC_NAME --action slide_check --out_file $DST_NAME --use_llm_transform --limit 0:10
vimdiff $SRC_NAME process_slides.txt
```

# Reduce all slides
```
SRC_NAME=$(ls $DIR/lectures_source/Lesson04.2*); echo $SRC_NAME
DST_NAME=process_slides.txt
process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $DST_NAME --use_llm_transform --limit 0:10
vimdiff $SRC_NAME process_slides.txt
```

# Generate all the slides.

> process_lesson.py --lectures 0*:1* --class data605 --target pdf

# Count pages.

> find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print -exec mdls -name kMDItemNumberOfPages {} \;

// process_slides.py --in_file data605/lectures_source/Lesson02-Git_Data_Pipelines.txt --action slide_format_figures --out_file data605/lectures_source/Lesson02-Git_Data_Pipelines.txt --use_llm_transform
