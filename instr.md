helpers_root/dev_scripts_helpers/system_tools/create_links.py and stage_links.py

print only the basename of the files

Instead of printing
```

src_file                                                                    | dst_file                                                                                                                      | current_state | target_state |
--------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------ |
class_project/project_template/.dockerignore                                | -                                                                                                                             | missing       | -            |
-                                                                           | msml610/tutorials/L03_knowledge_representation/.ipynb_checkpoints/L03_01_entailment_implication_inference-checkpoint.ipynb    | extra         | -            |
-                                                                           | msml610/tutorials/L03_knowledge_representation/.ipynb_checkpoints/L03_01_entailment_implication_inference_utils-checkpoint.py | extra         | -            |
-                                                                           | msml610/tutorials/L03_knowledge_representation/.ipynb_checkpoints/L03_02_wumpus_world-checkpoint.ipynb                        | extra         | -            |
```

print

```
src_dir=class_project/project_template/
dst_dir=msml610/tutorials/L03_knowledge_representation/

src_file                                                                    | dst_file                                                                                                                      | current_state | target_state |
--------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------ |
.dockerignore                                | -                                                                                                                             | missing       | -            |
-                                                                           | L03_01_entailment_implication_inference-checkpoint.ipynb    | extra         | -            |
-                                                                           | L03_01_entailment_implication_inference_utils-checkpoint.py | extra         | -            |
-                                                                           | L03_02_wumpus_world-checkpoint.ipynb                        | extra         | -            |
```
