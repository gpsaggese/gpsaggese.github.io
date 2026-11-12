# Generate book

  > vi book_springer/lectures_source/Lesson02.1*.smd book_springer/book/Lesson02.1*.tex

- Render the slides
  ```
  > gen_slides.py -i book_springer/02.1 --daemon
  ```

- Render the book
  ```
  > run_latex.py -i book_springer/book/book.tex --num_passes 3

  > run_latex.py -i book_springer/book/book.tex --daemon
  ```

  > class_scripts/create_book_toc_from_slides.py --max_level 2

- 

```
claude> /slides.add_references XYZ.smd

slides.criticize
slides.add_visuals
slides.review
```
