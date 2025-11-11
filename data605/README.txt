# Generate the lecture script

- Run from inside a container
  ```bash
  > i docker_bash --base-image=623860924167.dkr.ecr.eu-north-1.amazonaws.com/cmamp --skip-pull

  docker> sudo /bin/bash -c "(source /venv/bin/activate; pip install --upgrade openai)"

  docker> generate_slide_script.py \
    --in_file data605/lectures_source/Lesson01-Intro.txt \
    --out_file data605/lectures_source/Lesson01-Intro.script.txt \
    --slides_per_group 3 \
    --limit 1:5
  ```

- Run from outside the container
  ```bash
  > gen_data605_script.sh 04.3
  ```

# Check correctness of all the slides

- Run for one lecture from inside the container
  ```
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson02*); echo $SRC_NAME
  > DST_NAME=process_slides.txt
  docker> process_slides.py --in_file $SRC_NAME --action text_check --out_file $DST_NAME --use_llm_transform --limit 0:10
  > vimdiff $SRC_NAME process_slides.txt
  ```

- Run for one lecture outside the container
  ```
  > slide_check.sh 01.2
  ```

- Run for several lessons
  ```
  > process_lessons.py --lectures 01.1* --class data605 --action slide_check --limit 0:2
  ```

# Reduce all slides
```
SRC_NAME=$(ls $DIR/lectures_source/Lesson04.2*); echo $SRC_NAME
process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $SRC_NAME --use_llm_transform --limit 0:10
```

```
> slide_reduce.sh 01.1*
```

# Generate all the slides

> process_lessons.py --lectures 0*:1* --class data605 --action pdf

# Count pages

> find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print -exec mdls -name kMDItemNumberOfPages {} \;

> find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print0 | while IFS= read -r -d '' file; do     pages=$(mdls -name kMDItemNumberOfPages "$file" | awk -F'= ' '{print $2}');     echo -e "${file}\t${pages}"; done | tee tmp.txt

data605/lectures/Lesson01.1-Intro.pdf   10
data605/lectures/Lesson01.2-Big_Data.pdf        17
data605/lectures/Lesson01.3-Is_Data_Science_Just_Hype.pdf       14

> count_pages.sh | pbcopy

// process_slides.py --in_file data605/lectures_source/Lesson02-Git_Data_Pipelines.txt --action slide_format_figures --out_file data605/lectures_source/Lesson02-Git_Data_Pipelines.txt --use_llm_transform
