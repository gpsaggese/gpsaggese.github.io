if [[ 0 == 1 ]]; then
    ck_marketing/plugins/pitchbook/parse_mhtml.py --input_file '/Users/saggese/Desktop/People Results - People Screener_1.mhtml' --mode grids --table_index 0 --output_file output/out1.csv
    ck_marketing/plugins/pitchbook/parse_mhtml.py --input_file '/Users/saggese/Desktop/People Results - People Screener_2.mhtml' --mode grids --table_index 0 --output_file output/out2.csv
    ck_marketing/plugins/pitchbook/parse_mhtml.py --input_file '/Users/saggese/Desktop/People Results - People Screener_3.mhtml' --mode grids --table_index 0 --output_file output/out3.csv

    ck_marketing/plugins/pitchbook/merge_mhtml_csv_files.py --input_file output/out1.csv --input_file output/out2.csv --input_file output/out3.csv --output_file output/merged.csv
else
    DIR="fort500.corp_dev"
    # fort500.product_tech
    # top200AI.corp_dev
    # top200AI.product_tech
    ck_marketing/plugins/pitchbook/parse_mhtml.py --input_dir "/Users/saggese/Desktop/$DIR" --mode grids --table_index 0 --output_dir output

    ck_marketing/plugins/pitchbook/merge_mhtml_csv_files.py --input_dir output --output_file merged.csv
fi;

csvformat -T merged.csv | pbcopy
