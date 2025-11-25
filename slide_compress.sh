FILES="
data605/lectures_source/images/lecture_11_2/lec_11_2_slide_17_image_3.png
data605/lectures_source/images/lecture_11_1/lec_11_1_slide_7_image_1.png
data605/lectures_source/images/lecture_10_2/lec_10_2_slide_4_image_3.png
data605/lectures_source/images/lecture_11_1/lec_11_1_slide_17_image_3.png
data605/lectures_source/images/lecture_10_2/lec_10_2_slide_7_image_3.png
data605/lectures_source/images/lecture_11_2/lec_11_2_slide_17_image_2.png
data605/lectures_source/images/lecture_10_2/lec_10_2_slide_4_image_1.png
data605/lectures_source/images/lecture_2/lec_2_slide_50_image_1.png
data605/lectures_source/images/lecture_11_1/lec_11_1_slide_15_image_3.png
data605/lectures_source/images/lecture_11_1/lec_11_1_slide_17_image_2.png
data605/lectures_source/images/lecture_11_1/lec_11_1_slide_18_image_1.png
data605/lectures_source/images/lecture_4_2/lec_4_2_slide_4_image_1.png
data605/lectures_source/images/lecture_10_1/lec_10_1_slide_3_image_1.png
data605/lectures_source/images/lecture_2/lec_2_slide_42_image_1.png
data605/lectures_source/images/lecture_11_1/lec_11_1_slide_17_image_1.png
"
FILES=$(find data605/lectures_source/images -type f -exec du -h {} + | sort -rh | head -70)
mkdir -p compressed
for f in $FILES; do
    dst=compressed/$(basename $f)
    #pngquant --quality=65-80 --speed 1 --output $dst --force $f
    echo $f
    #magick $f -resize 600x600\> -background none -extent 600x600 -strip -quality 85 $f
    magick $f -resize 800x -strip -quality 85 $f
done
