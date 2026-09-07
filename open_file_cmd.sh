/usr/bin/osascript << EOF
set theFile to POSIX file "/Users/saggese/src/umd_classes2/msml610/book/Lesson03.2-Propositional_and_first_order_logic.pdf" as alias
tell application "Skim"
activate
set theDocs to get documents whose path is (get POSIX path of theFile)
if (count of theDocs) > 0 then revert theDocs
open theFile
end tell
EOF
