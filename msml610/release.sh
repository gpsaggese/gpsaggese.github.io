#!/bin/bash -xe
gen_slides.py -f "$(ls -1 msml610/lectures_source/*.smd)" --action release
render_book_chapter.py -f "$(ls -1 msml610/book/*.typ)" --action release
