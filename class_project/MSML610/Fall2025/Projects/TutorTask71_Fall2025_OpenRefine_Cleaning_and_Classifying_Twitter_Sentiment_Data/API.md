# API for Data Cleaning (OpenRefine)

## 1) Create an OpenRefine project
- Import the raw CSV: `training.1600000.processed.noemoticon.csv`
- Columns expected: `target,id,date,flag,user,text`

## 2) Apply the recipe
Open **Undo/Redo → Apply…** and paste the JSON below:

```json
[
  { "op":"core/column-rename", "oldColumnName":"Column 1", "newColumnName":"target" },
  { "op":"core/column-rename", "oldColumnName":"Column 2", "newColumnName":"id" },
  { "op":"core/column-rename", "oldColumnName":"Column 3", "newColumnName":"date" },
  { "op":"core/column-rename", "oldColumnName":"Column 4", "newColumnName":"flag" },
  { "op":"core/column-rename", "oldColumnName":"Column 5", "newColumnName":"user" },
  { "op":"core/column-rename", "oldColumnName":"Column 6", "newColumnName":"text" },

  {
    "op": "core/column-addition",
    "description": "Create text_clean from text (lowercase, remove urls/mentions/hashtags/non-letters, collapse spaces, trim)",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "text_clean",
    "columnInsertIndex": 0,
    "baseColumnName": "text",
    "expression": "value.toLowercase()\n.replace(/https?:\\S+/,\"\")\n.replace(/@\\S+/,\"\")\n.replace(/#/,\"\")\n.replace(/[^a-z\\s]/,\"\")\n.replace(/\\s+/,\" \")\n.trim()",
    "onError": "set-to-blank"
  },

  {
    "op": "core/row-removal",
    "description": "Remove rows where text_clean is blank after cleaning",
    "engineConfig": {
      "mode": "row-based",
      "facets": [
        {
          "type": "list",
          "name": "Blank text_clean",
          "columnName": "text_clean",
          "expression": "isBlank(value)",
          "selectBlank": true,
          "omitBlank": false,
          "selection": [ { "v": { "v": "true", "l": "true" } } ],
          "invert": false
        }
      ]
    }
  },

  {
    "op": "core/column-addition",
    "description": "word_count from text_clean",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "word_count",
    "columnInsertIndex": 1,
    "baseColumnName": "text_clean",
    "expression": "with(value.trim(), s, if(s==\"\", 0, s.split(/\\s+/).length()))",
    "onError": "set-to-blank"
  },
  {
    "op": "core/column-addition",
    "description": "char_count (no spaces) from text_clean",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "char_count",
    "columnInsertIndex": 2,
    "baseColumnName": "text_clean",
    "expression": "length(value.replace(/\\s+/,\"\"))",
    "onError": "set-to-blank"
  },
  {
    "op": "core/column-addition",
    "description": "avg_word_length = chars(without spaces)/words",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "avg_word_length",
    "columnInsertIndex": 3,
    "baseColumnName": "text_clean",
    "expression": "with(value.trim(), s, if(s==\"\", null, length(s.replace(/\\s+/,\"\")) / s.split(/\\s+/).length()))",
    "onError": "set-to-blank"
  },

  {
    "op": "core/column-addition",
    "description": "label_str from target (0/2/4 → negative/neutral/positive)",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "label_str",
    "columnInsertIndex": 4,
    "baseColumnName": "target",
    "expression": "if(value==0,'negative', if(value==2,'neutral', if(value==4,'positive','unknown')))",
    "onError": "set-to-blank"
  },

  {
    "op": "core/column-addition",
    "description": "pos_word_count using space-padded split counting on common positive tokens",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "pos_word_count",
    "columnInsertIndex": 5,
    "baseColumnName": "text_clean",
    "expression": "with(\" \" + value.toLowercase().replace(/\\s+/,\" \").trim() + \" \", s,\n  (s.split(\" love \").length()-1) +\n  (s.split(\" great \").length()-1) +\n  (s.split(\" awesome \").length()-1) +\n  (s.split(\" amazing \").length()-1) +\n  (s.split(\" excellent \").length()-1) +\n  (s.split(\" fantastic \").length()-1) +\n  (s.split(\" wonderful \").length()-1) +\n  (s.split(\" happy \").length()-1) +\n  (s.split(\" best \").length()-1) +\n  (s.split(\" nice \").length()-1) +\n  (s.split(\" cool \").length()-1) +\n  (s.split(\" fun \").length()-1) +\n  (s.split(\" yay \").length()-1) +\n  (s.split(\" excited \").length()-1) +\n  (s.split(\" proud \").length()-1) +\n  (s.split(\" wow \").length()-1) +\n  (s.split(\" enjoy \").length()-1) +\n  (s.split(\" enjoying \").length()-1) +\n  (s.split(\" lovely \").length()-1) +\n  (s.split(\" congrats \").length()-1) +\n  (s.split(\" thanks \").length()-1) +\n  (s.split(\" thankful \").length()-1) +\n  (s.split(\" grateful \").length()-1) +\n  (s.split(\" blessed \").length()-1) +\n  (s.split(\" haha \").length()-1) +\n  (s.split(\" hehe \").length()-1) +\n  (s.split(\" lol \").length()-1)\n)",
    "onError": "set-to-blank"
  },

  {
    "op": "core/column-addition",
    "description": "neg_word_count using space-padded split counting on common negative tokens",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "neg_word_count",
    "columnInsertIndex": 6,
    "baseColumnName": "text_clean",
    "expression": "with(\" \" + value.toLowercase().replace(/\\s+/,\" \").trim() + \" \", s,\n  (s.split(\" bad \").length()-1) +\n  (s.split(\" sad \").length()-1) +\n  (s.split(\" hate \").length()-1) +\n  (s.split(\" hated \").length()-1) +\n  (s.split(\" upset \").length()-1) +\n  (s.split(\" tired \").length()-1) +\n  (s.split(\" sick \").length()-1) +\n  (s.split(\" bored \").length()-1) +\n  (s.split(\" boring \").length()-1) +\n  (s.split(\" angry \").length()-1) +\n  (s.split(\" mad \").length()-1) +\n  (s.split(\" worst \").length()-1) +\n  (s.split(\" terrible \").length()-1) +\n  (s.split(\" awful \").length()-1) +\n  (s.split(\" annoying \").length()-1) +\n  (s.split(\" disappointed \").length()-1) +\n  (s.split(\" depressing \").length()-1) +\n  (s.split(\" depressed \").length()-1) +\n  (s.split(\" lonely \").length()-1) +\n  (s.split(\" miss \").length()-1) +\n  (s.split(\" ugh \").length()-1) +\n  (s.split(\" hurt \").length()-1) +\n  (s.split(\" broken \").length()-1) +\n  (s.split(\" sucks \").length()-1) +\n  (s.split(\" cry \").length()-1) +\n  (s.split(\" crying \").length()-1) +\n  (s.split(\" blah \").length()-1)\n)",
    "onError": "set-to-blank"
  },

  {
    "op": "core/column-addition",
    "description": "sentiment_score_norm = (pos-neg)/(pos+neg), with zero-guard",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "sentiment_score_norm",
    "columnInsertIndex": 7,
    "baseColumnName": "pos_word_count",
    "expression": "with(cells[\"pos_word_count\"].value, p, with(cells[\"neg_word_count\"].value, n, if((p+n)==0, 0, (p-n)/(p+n))))",
    "onError": "set-to-blank"
  },

  { "op":"core/text-transform", "engineConfig":{"facets":[],"mode":"row-based"}, "columnName":"word_count", "expression":"value.toNumber()", "onError":"keep-original", "repeat":false, "repeatCount":1, "description":"To number: word_count" },
  { "op":"core/text-transform", "engineConfig":{"facets":[],"mode":"row-based"}, "columnName":"char_count", "expression":"value.toNumber()", "onError":"keep-original", "repeat":false, "repeatCount":1, "description":"To number: char_count" },
  { "op":"core/text-transform", "engineConfig":{"facets":[],"mode":"row-based"}, "columnName":"avg_word_length", "expression":"value.toNumber()", "onError":"keep-original", "repeat":false, "repeatCount":1, "description":"To number: avg_word_length" },
  { "op":"core/text-transform", "engineConfig":{"facets":[],"mode":"row-based"}, "columnName":"pos_word_count", "expression":"value.toNumber()", "onError":"keep-original", "repeat":false, "repeatCount":1, "description":"To number: pos_word_count" },
  { "op":"core/text-transform", "engineConfig":{"facets":[],"mode":"row-based"}, "columnName":"neg_word_count", "expression":"value.toNumber()", "onError":"keep-original", "repeat":false, "repeatCount":1, "description":"To number: neg_word_count" },
  { "op":"core/text-transform", "engineConfig":{"facets":[],"mode":"row-based"}, "columnName":"sentiment_score_norm", "expression":"value.toNumber()", "onError":"keep-original", "repeat":false, "repeatCount":1, "description":"To number: sentiment_score_norm" }
]