- The world can have many states (Rain and WetGround)
- Model is a state of the world (i.e., an assignment of the variables)
- A sentence 

Sentence = logical statement built from variables, evaluates True or False in given model.

Sentence fixed. Model varies. Evaluate sentence's truth per model, that's satisfaction.

KB is a set of sentences that represent our understanding of the world

# Changes

http://localhost:8888/lab/tree/git_root/msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.ipynb

1) Always use bullet points in markdown

Do not do this

_Model table_: the same 4-row table, with $M(KB)$ shaded blue and
$M(\alpha)$ outlined in dashed orange
_Inclusion counts_: bar chart of $|M(KB)|$, the overlap with $M(\alpha)$,
and the counterexample rows
_Comments_: which `KB` sentences are toggled on, the query $\alpha$, and
the entailment verdict

but

- _Model table_: the same 4-row table, with $M(KB)$ shaded blue and $M(\alpha)$ outlined in dashed orange
- _Inclusion counts_: bar chart of $|M(KB)|$, the overlap with $M(\alpha)$, and the counterexample rows
- _Comments_: which `KB` sentences are toggled on, the query $\alpha$, and the entailment verdict


2) In cell1_1_models_and_satisfaction, remove model count

3) In cell2_1_entailment_model_checking, remove model inclusion

4) Add a KB to clarify that 
Rain
Rain => WetGround

are the knowledge base

5) Replace

Rain / Rain => WetGround

with 

KB: "Rain", "Rain => WetGround"


