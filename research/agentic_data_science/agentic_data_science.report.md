# Summary

- **Agentic data science**: LLM-driven systems that plan, use tools, write and
  execute code, inspect outputs, revise hypotheses, and produce analyst-grade
  deliverables across the data science lifecycle
- Accelerated in 2025-2026 due to new benchmarks (DSBench, MLE-bench, DARE-bench,
  AgentDS), focus on training data and memory, commercial adoption, and specialized
  agents
- **Key conclusion**: Agentic systems are powerful accelerators but do not yet
  replace senior data scientists, especially for problem formulation, domain
  reasoning, and decision-making

# 1. What Is Agentic Data Science?

- Traditional AI analytics tools:
  - NL-to-SQL
  - Code autocomplete
  - Chart generation
  - AutoML
- Agentic data science systems instead perform multi-step autonomous workflows,
  including:
  1. Task decomposition
  2. Tool selection
  3. Data access
  4. Code generation
  5. Execution
  6. Error handling
  7. Iteration
  8. Modeling
  9. Evaluation
  10. Report writing
- **Traditional analytics AI**: Answers questions
- **Agentic data science**: Conducts investigations
- This represents a fundamental shift from query answering to analytical reasoning
  systems

# 2. The Data Science Lifecycle and Agents

- A full data science workflow includes:
  1. Business understanding
  2. Data acquisition
  3. Data cleaning
  4. Exploratory analysis
  5. Feature engineering
  6. Modeling
  7. Evaluation
  8. Interpretation
  9. Deployment
  10. Monitoring
- Most current agent systems are strong in:
  - Data exploration
  - Visualization
  - Modeling
  - Experimentation
  - Report generation
- They are weaker in:
  - Problem framing
  - Causal inference
  - Business strategy
  - Production ownership
  - Monitoring and governance

# 3. Research Architecture of Agentic Systems

- Modern agentic data science systems typically include:
  - **Planner**: Breaks tasks into steps
  - **Tool Layer**: SQL, Python, Pandas, Spark, visualization tools, ML libraries,
    APIs, retrieval systems
  - **Execution Sandbox**: Runs generated code safely
  - **Memory**: Stores previous queries, intermediate results, learned patterns, past
    failures
  - **Evaluator / Critic**: Checks errors, metrics, constraints, data leakage, model
    performance
  - **Report Generator**: Produces charts, insights, narrative, recommendations

# 4. Research Trend: Agents as Search Systems

- **Key insight**: Data science agents are not just chatbots, they are search systems
  over possible analyses
- Agents explore:
  - Different models
  - Different features
  - Different transformations
  - Different hyperparameters
  - Different evaluation metrics
  - Different hypotheses
This makes agentic data science closer to:
- AutoML
- Bayesian optimization
- Experiment search
- Scientific discovery systems
Than to chatbots

# 5. Benchmark Landscape (Important)

- Benchmarks show what the field actually measures
- **DSBench**: Measures realistic data science tasks (data analysis, data modeling,
  Kaggle-style problems)
- **MLE-bench**: Evaluates agents on Kaggle competitions (read problem, explore data,
  train models, iterate, submit). Measures whether agents can achieve **Kaggle
  medal-level performance**
- **DARE-bench**: Large benchmark with thousands of modeling tasks. Focuses on
  verifiable results rather than subjective evaluation
- **AgentDS**: Measures human + AI collaboration vs AI alone across industries
  Conclusion: Human expertise still critical

# 6. Current Capabilities of Agentic Data Science

## Strong Areas

- Agents are currently very good at:
  - **NL -> SQL Analytics**: Example - "What were sales by region last quarter?"
  - **Exploratory Data Analysis**: Summary statistics, correlations, visualizations,
    outlier detection
  - **Feature Engineering**: Encoding, scaling, feature creation, interaction terms
  - **Model Training**: Regression, classification, tree models, neural networks,
    hyperparameter tuning
  - **Experiment Iteration**: Try multiple models, compare metrics, select best model
  - **Report Generation**: Charts, written insights, executive summaries

# 7. Where Agents Still Fail

## Major Limitations

- **Problem Formulation**: Agents struggle to convert vague business questions into
  analytical problems
  - Example "Why are customers leaving?" requires cohort analysis, survival analysis,
    causal inference, business understanding
- **Causal Reasoning**: Agents mostly do correlation and prediction, not causality
- **Data Leakage Detection**: Still a major issue in automated modeling
- **Metric Selection**: Agents may optimize wrong metrics
- **Business Decision Context**: Agents do not understand market conditions,
  strategy, risk tolerance, organizational constraints
- **Deployment & Monitoring**: Production ML lifecycle still human-heavy

# 8. Commercial Landscape

- Market is converging into three major categories:
  - **Warehouse Analytics Agents**: Embedded inside data warehouses
    - Use cases: ask questions about data, generate dashboards, write SQL
      automatically
  - **Lakehouse / ML Platform Agents**: Integrated with data pipelines, notebooks, ML
    workflows
    - Use cases: build models, run experiments, create pipelines, automate workflows
  - **Analytics Workbench Agents**: Notebook-style environments with AI analysts
    - Use cases: EDA, visualization, analysis notebooks, reporting, collaboration

# 9. Emerging Architecture Pattern (Very Important)

- Strongest agentic data science systems follow this flow:
  - **User Question**: Input from the user
  - **Task Planner**: Breaks down the question
  - **Semantic Layer / Data Catalog**: Maps to available data
  - **Tool Selection**: Chooses SQL / Python / ML approach
  - **Execution Sandbox**: Runs the code safely
  - **Evaluation / Critic**: Checks quality and errors
  - **Memory Update**: Stores results and learnings
  - **Iteration Loop**: Refines if needed
  - **Report Generator**: Creates final output
- This loop is the core architecture of agentic analytics systems

# 10. Risks in Agentic Data Science

## Major Risks

- **Silent Analytical Errors**: Agent produces convincing but wrong analysis
- **Governance Issues**: Agents may use incorrect metrics or unauthorized data
- **Security / Privacy**: Agents accessing multiple data systems
- **Over-Automation**: Organizations trusting AI analysis without validation
- **Compute Cost**: Long-horizon agent search can be expensive
- **Benchmark Overfitting**: Success on Kaggle-style tasks does not equal real
  business value

# 11. Strategic Outlook (2026-2028)

## Likely Industry Evolution

- **Short Term (1-2 Years)**:
  - NL analytics agents become standard
  - AI-generated dashboards
  - Automated exploratory data analysis (EDA)
  - AI experiment assistants
- **Medium Term (3-5 Years)**:
  - Autonomous experiment loops
  - Automated feature stores
  - Self-improving machine learning (ML) pipelines
  - AI research assistants
  - Data science copilots become default
- **Long Term (5-10 Years)**:
  - Autonomous analytics systems
  - Continuous decision optimization
  - AI-run experimentation platforms
  - AI-driven companies

# 12. Key Insight: the Role of Humans Will Change

- Future workflow will likely be:
| Step           | Human | Agent |
| -------------- | ----- | ----- |
| Define problem | ✓     |       |
| Access data    |       | ✓     |
| Clean data     |       | ✓     |
| Explore data   |       | ✓     |
| Build models   |       | ✓     |
| Evaluate       | ✓     | ✓     |
| Interpret      | ✓     |       |
| Decide         | ✓     |       |
| Deploy         | ✓     | ✓     |
| Monitor        | ✓     | ✓     |
- **Conclusion:**
  - AI will automate **analysis**, but humans will still own **decisions**

## Most Important Points

- Agentic data science is real and growing fast
- Biggest improvement is in multi-step analytical workflows
- Benchmarks now measure full data science tasks, not just prompts
- Specialized data-science-trained agents are emerging
- Warehouse-native agents will dominate near-term adoption
- Fully autonomous data scientists are still far away
- Future is human + AI collaborative analytics
- Biggest value is productivity and experiment speed
- Governance and semantic layers will be critical infrastructure
- Winning companies will own the **data + context + execution environment**, not just
  the model