#!/bin/bash
DIR=lectures_source
FILES=$(ls -1 $DIR/Lesson*)

for FILE in $FILES; do
    if [[ 1 == 0 ]]; then
        echo "##########################################################################"
        echo "# $FILE"
        echo "##########################################################################"
        VAL=$(grep "^*" $FILE | wc -l)
        echo "  Number of slides: $VAL"
        VAL=$(grep "^\- " $FILE | wc -l)
        echo "  Number of first level bullets: $VAL"
        VAL=$(grep "\- " $FILE | wc -l)
        echo "  Number of all level bullets: $VAL"

        \grep "^#" $FILE | \grep -v "######"
    else
        VAL=$(grep "^*" $FILE | wc -l)
        echo "$FILE	$VAL"
    fi;
done

# Total
if [[ 1 == 0 ]]; then
    NUM_LESSONS=$(ls -1 $DIR/Lesson* | wc -l)
    echo "# Number of lessons: $NUM_LESSONS"
    VAL=$(grep -E "\*" $FILES | wc -l)
    echo "  Number of slides: $VAL"
    VAL=$(grep "^\- " $FILES | wc -l)
    echo "  Number of first level bullets: $VAL"
    VAL=$(grep "\- " $FILES | wc -l)
    echo "  Number of all level bullets: $VAL"
fi;

# Probably we can do 20 slides per hour, so around 50 slides per class

# python -c "print(1306 * 3 / 60 / 3)"
# 21.766666666666666
