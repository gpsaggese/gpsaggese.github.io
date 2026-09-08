/usr/bin/osascript << EOF
set theFile to POSIX file "/Users/saggese/src/umd_classes1/msml610/book.tmp/Lesson02.5-ML_Techniques_Model_Evaluation.pdf" as alias
tell application "Skim"
activate
set theDocs to get documents whose path is (get POSIX path of theFile)
if (count of theDocs) > 0 then revert theDocs
open theFile
end tell
EOF
