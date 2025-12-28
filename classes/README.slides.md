# DATA605

## Check correctness of all the slides

- Check one lecture from inside the container
  ```
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson02*); echo $SRC_NAME
  > DST_NAME=process_slides.txt
  docker> process_slides.py --in_file $SRC_NAME --action text_check --out_file $DST_NAME --use_llm_transform --limit 0:10
  > vimdiff $SRC_NAME process_slides.txt
  ```

- Check one lecture outside the container
  ```
  > slide_check.sh 01.2
  ```

- Check several lessons
  ```
  > process_lessons.py --lectures 01.1* --class data605 --action slide_check --limit 0:2
  ```

## Improve slides

- Run
  ```
  > llm_transform.py -i data605/lectures_source/Lesson07.2-Data_Wrangling.txt -p slide_improve -v DEBUG
  ```

## Reduce all slides

- Run
  ```bash
  > SRC_NAME=$(ls $DIR/lectures_source/Lesson04.2*); echo $SRC_NAME
  > process_slides.py --in_file $SRC_NAME --action slide_reduce --out_file $SRC_NAME --use_llm_transform --limit 0:10
  ```
  or
  ```
  > slide_reduce.sh 01.1*
  ```

# Generate the PDF for all the slides

- Run
  ```
  > process_lessons.py --lectures 0*:1* --class data605 --action pdf
  ```

## Generate the lecture script

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

- Generate the intro
  > TAG=08.3; llm_cli.py -i data605/lectures_script/Lesson${TAG}*.script.txt -p "You are a college professor and you need to do an introduction in 50 word the content of the slides starting with In this lesson" -o -

- Generate the outro
  > TAG=08.3; llm_cli.py -i data605/lectures_script/Lesson${TAG}*.script.txt -p "You are a college professor and you need to summarize what was discussed in less than 50 word in the slides like In this lesson we have discussed" -o -

## Count pages

> find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print -exec mdls -name kMDItemNumberOfPages {} \;

> find data605/lectures/Lesson0*.pdf -type f -name "*.pdf" -print0 | while IFS= read -r -d '' file; do     pages=$(mdls -name kMDItemNumberOfPages "$file" | awk -F'= ' '{print $2}');     echo -e "${file}\t${pages}"; done | tee tmp.txt

data605/lectures/Lesson01.1-Intro.pdf   10
data605/lectures/Lesson01.2-Big_Data.pdf        17
data605/lectures/Lesson01.3-Is_Data_Science_Just_Hype.pdf       14

> count_pages.sh | pbcopy

// process_slides.py --in_file data605/lectures_source/Lesson02-Git_Data_Pipelines.txt --action slide_format_figures --out_file data605/lectures_source/Lesson02-Git_Data_Pipelines.txt --use_llm_transform

## Count words

> dir="data605/lectures_script/"; for f in "$dir"/*; do [ -f "$f" ] && printf "%s\t%s\n" "$(basename "$f")" "$(wc -w < "$f")"; done

./count_words.sh

## Review scripts

> TAG=10.1; gen_data605.sh $TAG; vi $(ls data605/lectures_script/*${TAG}*)

> TAG=08.3; open data605/lectures/Lesson${TAG}*.pdf; vi $(ls data605/lectures_script/*${TAG}*)

# MSML610

> cd $GIT_ROOT

> notes_to_pdf.py --input data605/lectures_md/final_enhanced_markdown_lecture_2.txt --output tmp.pdf --type slides --skip_action cleanup_after --debug_on_error --toc_type navigation --filter_by_slides 1:4

> gen_data605.sh 01

> gen_msml610.sh 02

> FILE=msml610/lectures_source/Lesson05*
> process_slides.py --in_file $FILE --action slide_format_figures --out_file $FILE --use_llm_transform
> process_slides.py --in_file $FILE --action slide_check --out_file ${FILE}.check --use_llm_transform --limit None:10

> rsync -avz -e "ssh -i ~/.ssh/ck/saggese-cryptomatic.pem" saggese@$DEV1:/data/saggese/src/umd_classes1/msml610/lectures/ msml610/lectures/; open msml610/lectures/*07.1* -a "skim"

## To run the tutorials
> cd msml610/tutorials

> i docker_jupyter --skip-pull --stage local --version 1.0.0

> open -a "Chrome" http://127.0.0.1:5011/lab/tree/notebooks/Bayesian_Coin.ipynb

## Fix slides

> FILE=data605/lectures_source/Lesson09.2-Spark_Primitives.txt
> llm_cli.py --input $FILE -pf "fix_slides.prompt.md" -o improved.md --model "gpt-4o" -b
