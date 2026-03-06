You are given a directory containing lecture text files and image files.

Given the file log.txt

msml610/lectures_source/Lesson07.3-Hierarchical_Models.txt:86:![](msml610/lectures_source/figures/Lesson07_Pooled_unpooled_hierarchical_models.png){ width=80% }                                                                                                                            msml610/lectures_source/Lesson07.3-Hierarchical_Models.txt:101:![](msml610/lectures_source/figures/Lesson07_Chemical_shift_hierarchical1.png)
msml610/lectures_source/Lesson07.3-Hierarchical_Models.txt:133:![](msml610/lectures_source/figures/Lesson07_Chemical_shift_hierarchical2.png)

Root directory:
msml610/lectures_source/

Text files:
msml610/lectures_source/*.txt

Image files:
msml610/lectures_source/figures/*

The text files contain references to image files using markdown syntax such as:

![](msml610/lectures_source/figures/Lesson04_Kernel_Trick.png)

For every file:

msml610/lectures_source/*.txt

1. Extract every referenced image path of the form

msml610/lectures_source/figures/<filename>

2. Ignore image files if:

- The filename is exactly:
  UMD_Logo.png

- The filename starts with:
  Book_cover_

3. Determine the lesson identifier from the TXT filename.

Example TXT filename:

Lesson01.3-Brief_History_of_AI.txt

The lesson identifier is:

01.3

The prefix to use in the renamed figure is:

L01.3

4. For each referenced image file:

Original format example:

Lesson04_Kernel_Trick.png

Remove the leading `LessonXX` prefix and keep only the descriptive part.

Examples:

Lesson04_Kernel_Trick.png
→ Kernel_Trick.png

Lesson07_Multiple_linear_regression1.png
→ Multiple_linear_regression1.png

Lesson94_Entropy_vs_Variance.png
→ Entropy_vs_Variance.png

5. Construct the new filename using the format:

L<lesson_id>.<name>

Example:

Lesson01.Windows_failure.png
→ L01.3.Windows_failure.png

Full path example:

msml610/lectures_source/figures/L01.3.Windows_failure.png

6. Output a CSV-style mapping line for every figure reference:

format:

<txt_file>,<original_figure_path>,<new_figure_path>

Example:

msml610/lectures_source/Lesson01.3-Brief_History_of_AI.txt,
msml610/lectures_source/figures/Lesson01.Windows_failure.png,
msml610/lectures_source/figures/L01.3.Windows_failure.png

Notes:

• If the same image appears multiple times, output the mapping only once.  
• Preserve the file extension (.png, .jpg, etc.).  
• Ignore any markdown formatting such as `{width=40%}` or `{height=50%}`.

------------------------------------------------------------
EXAMPLES OF IMAGE REFERENCES
------------------------------------------------------------

Examples that appear inside TXT files:

![](msml610/lectures_source/figures/Lesson07_Multiple_linear_regression1.png)

![](msml610/lectures_source/figures/Lesson07_Beta_distribution.png)

![](msml610/lectures_source/figures/Lesson04_Kernel_Trick.png)

![](msml610/lectures_source/figures/Lesson01_Ex_machina.jpg)

Your solution must correctly parse all such occurrences.


