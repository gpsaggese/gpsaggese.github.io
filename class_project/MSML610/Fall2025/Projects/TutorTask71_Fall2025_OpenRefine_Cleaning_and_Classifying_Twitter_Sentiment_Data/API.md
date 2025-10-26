# API for Data Cleaning (OpenRefine)

## 1) Create an OpenRefine project
- Import the raw CSV: `training.1600000.processed.noemoticon.csv`
- Columns expected: `target,id,date,flag,user,text`

## 2) Apply the recipe
Open **Undo/Redo → Apply…** and paste the JSON below:

```json
[
  {
    "op": "core/column-addition",
    "description": "Create text_clean from text (lowercase, remove urls/mentions/hashtags/specials, collapse spaces, trim)",
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
    "description": "Add word_count from text_clean",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "word_count",
    "columnInsertIndex": 1,
    "baseColumnName": "text_clean",
    "expression": "isBlank(value) ? 0 : value.split(\" \").length()",
    "onError": "set-to-blank"
  },
  {
    "op": "core/column-addition",
    "description": "Add char_count from text_clean",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "char_count",
    "columnInsertIndex": 2,
    "baseColumnName": "text_clean",
    "expression": "isBlank(value) ? 0 : value.length()",
    "onError": "set-to-blank"
  },
  {
    "op": "core/column-addition",
    "description": "Create label_str from target (0/2/4 → negative/neutral/positive)",
    "engineConfig": { "mode": "row-based" },
    "newColumnName": "label_str",
    "columnInsertIndex": 3,
    "baseColumnName": "target",
    "expression": "if(value==0,'negative', if(value==2,'neutral', if(value==4,'positive','unknown')))",
    "onError": "set-to-blank"
  }
]
