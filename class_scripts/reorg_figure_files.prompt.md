Process the content in log.txt

For each file in log.txt find which file is used in

- Create a map from the name of the figure file to a format like
  msml610/lectures_source/figures/L<XY.Z>.<name>.png
  - <name> should not contain Lesson and the number but only the text (e.g.,
    Kernel_Trick.png and not Lesson04_Kernel_Trick.png)

<example1>
Inside msml610/lectures_source/Lesson01.3-Brief_History_of_AI.txt

there is a file

msml610/lectures_source/figures/Lesson01.Windows_failure.png

Since the file storing the image is Lesson01.3-Brief_History_of_AI.txt and 
thus Lesson01.3, then L{XY.Z} should be L01.3
so the full file name should be like
msml610/lectures_source/figures/L01.3.Windows_failure.png
<example1>

- Print the output as a mapping
  txt file, original file, modified file

  msml610/lectures_source/Lesson01.3-Brief_History_of_AI.txt,msml610/lectures_source/figures/Lesson01.Windows_failure.png,msml610/lectures_source/figures/L01.3.Windows_failure.png,
