# Infer DAGs from Papers Milestones

- This file maps each version of the project to a short description and to the
  requirements of
  [`research/Infer_DAGs_from_papers/docs/01-prd.md`](/research/Infer_DAGs_from_papers/docs/01-prd.md)
  that it implements
- Releases are cumulative: a version also has the features of the versions before it
- The scope of each version is also in
  [Scope by Version](/research/Infer_DAGs_from_papers/docs/01-prd.md#scope-by-version)

## v1.0: DAG of a Single Paper

- **Description**: from the markdown of each paper, build a causal DAG of that paper
  alone, and audit it with the data of the user
  - The DAG uses only the text of the paper: no other paper, no knowledge of an LLM,
    no metadata of a dataset
- **Deliverable**: the user runs the pipeline on a list of papers and gets, for each
  paper, the records, the DAG, the review report, the reviewed DAG, the final graph,
  the audit report, and the pictures
- **Requirements**:
  - Input
    - [`R1`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r1-input-papers):
      read the markdown file of each paper
  - Extract relationships
    - [`R2`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r2-relationship-record):
      one record for each relationship
    - [`R3`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r3-provenance-pointer):
      quoted span checked in the markdown
    - [`R4`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r4-stated-and-inferred-status):
      `stated` or `inferred` status
    - [`R5`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r5-paper-without-relationships):
      empty DAG for a paper without relationships
  - Normalize variables
    - [`R6`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r6-canonical-id):
      canonical id and operator for each variable
    - [`R7`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r7-unknown-variable):
      unknown variable listed with a proposed id
    - [`R8`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r8-review-report):
      no approval gate, a review report to keep or remove nodes and edges
  - Build the DAG of a paper
    - [`R9`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r9-paper-dag):
      DAG of a paper, with no cycle
    - [`R10`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r10-unoriented-relationship):
      unoriented relationships
    - [`R11`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r11-feedback-loop):
      feedback loops
    - [`R25`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r25-reviewed-dag):
      reviewed DAG, without the nodes and edges that the user removes
  - Audit with data
    - [`R15`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r15-data-input):
      dataset and mapping to the columns
    - [`R16`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r16-conditional-independence-audit):
      conditional independence tests with false discovery rate control
    - [`R17`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r17-edge-verdict):
      verdict for each edge
    - [`R18`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r18-collider-bias):
      no edge from collider bias
    - [`R19`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r19-final-graph):
      final graph, separate from the DAG of the paper
  - Output
    - [`R20`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r20-files-for-each-paper):
      records, DAG, and review report files for each paper
    - [`R21`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r21-graph-and-report-files):
      final graph and audit report as JSON
    - [`R22`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r22-picture-of-a-graph):
      picture of a DAG
    - [`R23`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r23-issues-file):
      issues file for each stage
    - [`R24`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r24-rerun-skips-processed-papers):
      rerun skips processed papers and keeps the marks of the review report

## v2.0: Literature Graph

- **Description**: merge the DAGs of many papers into one literature graph, with the
  papers behind each edge and the conflicts between papers
- **Deliverable**: the user gets one literature graph for the list of papers, and the
  conflicts between papers
- **Requirements**:
  - New:
    - [`R12`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r12-literature-graph):
      merge the reviewed DAGs over canonical ids
    - [`R13`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r13-conflict-between-papers):
      keep every claim of conflicting papers
    - [`R14`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r14-opposite-directions):
      opposite directions keep the graph acyclic
  - Extended to the literature graph:
    - [`R15`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r15-data-input) to
      [`R19`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r19-final-graph): the
      audit with data runs on the literature graph
    - [`R22`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r22-picture-of-a-graph):
      the picture also marks the conflicting edges
    - [`R24`](/research/Infer_DAGs_from_papers/docs/01-prd.md#r24-rerun-skips-processed-papers):
      a rerun with new papers updates the literature graph

## v3.0: LLM Knowledge and Metadata

- **Description**: use the knowledge of an LLM and the metadata of a dataset, as DMCD
  does, in addition to the text of the papers
- **Deliverable**: not defined yet
- **Requirements**: not written yet, they come after v2.0
  - It relaxes the constraint that the relationships come from the text of the paper,
    see
    [Constraints and Assumptions](/research/Infer_DAGs_from_papers/docs/01-prd.md#constraints-and-assumptions)
  - Open question to answer first:
    [Open Questions](/research/Infer_DAGs_from_papers/docs/01-prd.md#open-questions),
    item 2 (can the `datamap` code run on variables derived from text?)
