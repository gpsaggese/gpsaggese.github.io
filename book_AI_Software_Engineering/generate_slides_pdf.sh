#!/bin/bash -xe
gen_slides.py \
  --files "$(ls book_AI_Software_Engineering/lectures_source/Lesson*.smd | xargs)" \
  --action release \
  --no_abort_on_error

# for_loop_lessons.py --class msml610 --lectures "01*:02*" --action release_slides_pdf
