> i docker_bash --base-image=623860924167.dkr.ecr.eu-north-1.amazonaws.com/cmamp --skip-pull

docker> sudo /bin/bash -c "(source /venv/bin/activate; pip install --upgrade openai)"

docker> generate_slide_script.py \
  --in_file data605/lectures_source/Lesson01-Intro.txt \
  --out_file data605/lectures_source/Lesson01-Intro.script.txt \
  --slides_per_group 3 \
  --limit 1:5

docker> gen_data605_script.sh 01 --limit 1:5
