#!/bin/bash -e
# > ls -1 lectures_source/Lesson*
FILES="
lectures_source/Lesson00-Class.txt
lectures_source/Lesson01-Intro.txt
"

#lectures_source/Lesson02-Techniques.txt
#lectures_source/Lesson03-Knowledge_representation.txt
#lectures_source/Lesson04-Models.txt
#lectures_source/Lesson05-Theories.txt
#lectures_source/Lesson06-Bayesian_statistics.txt
#lectures_source/Lesson07-Probabilistic_programming.txt
#lectures_source/Lesson08-Reasoning_over_time.txt
#lectures_source/Lesson09-Causal_inference.txt
#lectures_source/Lesson10-Timeseries_forecasting.txt
#lectures_source/Lesson11-Probabilistic_deep_learning.txt
#lectures_source/Lesson12-Reinforcement_learning.txt
#lectures_source/Lesson91.Refresher_probability.txt
#lectures_source/Lesson92.Refresher_probability_distributions.txt
#lectures_source/Lesson93.Refresher_linear_algebra.txt
#lectures_source/Lesson94.Refresher_information_theory.txt
#lectures_source/Lesson95.Refresher_game_theory.txt
#lectures_source/Lesson96.Refresher_stochastic_processes.txt
#lectures_source/Lesson97.Refresher_numerical_optimization.txt

#FILES="
#lectures_source/Lesson07-Probabilistic_programming.txt
#"

SRC_DIR="msml610"
DST_DIR="msml610/lectures"
if [[ 0 == 1 ]]; then
rm -rf $DST_DIR || true
mkdir $DST_DIR
fi;

for FILE in $FILES; do
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    rm -rf tmp* figs*
    echo $FILE
    cmd="notes_to_pdf.py --input $SRC_DIR/$FILE --output tmp.pdf --type slides --toc_type navigation --skip_action open"
    echo $cmd
    eval $cmd
    #
    ls $DST_DIR
    NEW_FILE=$(echo $FILE | sed 's/\.txt$/.pdf/')
    NEW_FILE=$(basename $NEW_FILE)
    cmd="mv tmp.pdf $DST_DIR/$NEW_FILE"
    echo $cmd
    eval $cmd
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
done

# cp -rf $DST_DIR/*.pdf ~/src/umd_data605_1/lectures_source/lectures
