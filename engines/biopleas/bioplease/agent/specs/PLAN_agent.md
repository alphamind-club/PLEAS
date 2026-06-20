# PLAN Agent Specification v2.0

## Overview
The PLAN agent is the first phase of the PLEAS (Plan-Learn-Execute-Assess-Share) framework. It decomposes complex biological/scientific tasks into structured, actionable sub-steps with clear success criteria, resource identification, and risk assessment.

---

## Purpose & Core Responsibilities

### Primary Purpose
Transform a user's high-level scientific query into a detailed, executable plan that can guide the downstream agents (LEARN, EXECUTE, ASSESS, SHARE).

### Core Responsibilities
1. **Task Decomposition**: Break complex queries into ordered, manageable sub-tasks
2. **Resource Identification**: Identify required tools, datasets, and software
3. **Success Criteria Definition**: Establish measurable outcomes for each step
4. **Risk Assessment**: Identify potential failure points and mitigation strategies
5. **Budget Awareness**: Estimate resource needs (time, cost, computational complexity)
6. **Constraint Recognition**: Acknowledge limitations and assumptions

---

## Input Contract

### Required Inputs
```python
{
    "user_query": str,              # The scientific task/question
    "available_tools": list[dict],  # Tool descriptions and schemas
    "data_lake": list[dict],        # Available datasets with descriptions
    "custom_uploads": list[dict],   # User-provided materials (PDFs, CSVs, etc.)
    "budget": {
        "time_seconds": int,        # Maximum execution time
        "cost_usd": float,          # Maximum cost
        "tokens": int               # Token budget
    }
}
```

### Optional Inputs
```python
{
    "custom_software": list[dict],  # Custom CLI tools or scripts
    "retrieval_context": str,       # Retrieved relevant information
    "previous_plan": str,           # If revising an existing plan
    "constraints": list[str]        # User-specified constraints
}
```

---

## Output Contract

### Required Outputs
```python
{
    "plan_steps": [
        {
            "step_id": int,
            "description": str,
            "checkbox_status": "[ ]" | "[✓]" | "[✗]",
            "required_tools": list[str],
            "required_data": list[str],
            "estimated_time_s": int,
            "estimated_cost_usd": float,
            "success_criteria": str,
            "dependencies": list[int]  # IDs of prerequisite steps
        }
    ],
    "assumptions": list[str],
    "risks": [
        {
            "risk": str,
            "severity": "low" | "medium" | "high",
            "mitigation": str
        }
    ],
    "resource_summary": {
        "total_estimated_time_s": int,
        "total_estimated_cost_usd": float,
        "critical_tools": list[str],
        "critical_data": list[str]
    }
}
```

### Output Format
Structured markdown with checklist:
```markdown
## Plan for: {task_title}

### High-Level Strategy
{1-2 paragraph overview}

### Steps
1. [ ] Step 1: {description}
   - Tools: {tool1, tool2}
   - Data: {dataset1}
   - Success: {criteria}
   - Est. Time: {N}s, Cost: ${X}

2. [ ] Step 2: {description}
   ...

### Assumptions
- Assumption 1
- Assumption 2

### Risks
- **High**: {risk} → Mitigation: {approach}
- **Medium**: {risk} → Mitigation: {approach}

### Resource Requirements
- Total Time: {N}s ({M}% of budget)
- Total Cost: ${X} ({Y}% of budget)
- Critical Tools: {list}
```

---

## Decision Logic & Heuristics

### Step Ordering Heuristics
1. **Data before computation**: Always plan data acquisition/validation before analysis
2. **Validation early**: Include validation steps after critical computations
3. **Checkpoints**: Insert assessment checkpoints every 3-5 steps
4. **Dependencies**: Respect tool/data dependencies (e.g., need sequence before alignment)

### Resource Estimation
```python
# Time estimation heuristics
def estimate_step_time(step_type):
    if step_type == "data_retrieval":
        return 30  # seconds
    elif step_type == "sequence_alignment":
        return 120
    elif step_type == "statistical_analysis":
        return 60
    elif step_type == "figure_generation":
        return 45
    # Default
    return 60
```

### Budget Allocation Strategy
- Reserve 20% buffer for unexpected issues
- Allocate more time to EXECUTE than other phases (50% of total)
- Front-load critical decisions (PLAN: 10%, LEARN: 15%, EXECUTE: 50%, ASSESS: 15%, SHARE: 10%)

### Tool Selection Criteria
1. **Relevance**: Tool description matches task requirements
2. **Availability**: Tool exists in registry and is accessible
3. **Reliability**: Prefer well-tested tools over experimental ones
4. **Efficiency**: Prefer faster tools when multiple options exist

### Risk Identification
**High-Risk Scenarios:**
- Required data not in data lake
- Tool requires manual installation
- Multi-step pipeline with no intermediate validation
- Budget insufficient for estimated work

**Mitigation Strategies:**
- Always suggest alternative approaches
- Include fallback options for critical steps
- Recommend validation/sanity checks

---

## Quality Criteria

### Clarity (0-10 scale)
- **10**: Every step is unambiguous, executable without clarification
- **7-9**: Minor ambiguities that experienced user can resolve
- **4-6**: Multiple steps need clarification
- **0-3**: Plan is confusing or contradictory

### Completeness (0-10 scale)
- **10**: All user requirements covered, no gaps
- **7-9**: Minor requirements missing but core task covered
- **4-6**: Significant requirements overlooked
- **0-3**: Major gaps in plan

### Feasibility (0-10 scale)
- **10**: All required resources available, realistic timeline
- **7-9**: Minor resource gaps or tight timeline
- **4-6**: Some resources unavailable or timeline unrealistic
- **0-3**: Plan cannot be executed with available resources

### Efficiency (0-10 scale)
- **10**: Optimal path, no unnecessary steps
- **7-9**: Minor inefficiencies but acceptable
- **4-6**: Notable redundancy or circuitous approach
- **0-3**: Highly inefficient or circular logic

### Overall Quality Score
```python
quality_score = (clarity + completeness + feasibility + efficiency) / 40.0
```

---

## Known Limitations

### Technical Limitations
1. **Cannot predict exact execution time**: Estimates are heuristic-based
2. **No runtime state**: Cannot adapt plan based on intermediate results (that's ASSESS's job)
3. **Limited to known tools**: Cannot invent new tools or methods
4. **Data lake scope**: Only knows about registered datasets, not external sources

### Contextual Limitations
1. **May over-specify for simple tasks**: Bias toward thoroughness
2. **Conservative estimates**: Tends to overestimate time/cost
3. **Assumes tools work as documented**: Cannot predict tool failures
4. **Limited domain knowledge**: Relies on tool descriptions and retrieval

### Scope Limitations
- **Does not execute**: PLAN only creates roadmap, doesn't run code
- **Does not retrieve data**: Identifies data needs but doesn't fetch them
- **Does not validate feasibility in real-time**: Assumptions may be incorrect

---

## Best Practices

### For Successful Planning
1. **Start broad, then narrow**: Begin with high-level goals, refine into specifics
2. **Think in phases**: Group related steps together
3. **Plan for failure**: Always include "what if" alternatives
4. **Document assumptions**: Make implicit knowledge explicit
5. **Include validation**: Don't assume outputs are correct
6. **Think like a project manager**: What needs to be done, in what order, with what resources

### For Specific Task Types

#### Experimental Design (e.g., CRISPR screens)
```markdown
1. [ ] Define biological question and hypothesis
2. [ ] Identify target gene set (literature + databases)
3. [ ] Design sgRNA library (tool: design_crispr_library)
4. [ ] Calculate experimental parameters (MOI, coverage)
5. [ ] Generate validation metrics (on-target score, off-target)
6. [ ] Output library and protocol
```

#### Data Analysis (e.g., RNA-seq)
```markdown
1. [ ] Validate input data format and quality
2. [ ] Align reads to reference genome
3. [ ] Quantify gene expression
4. [ ] Perform differential expression analysis
5. [ ] Generate visualizations (volcano plot, heatmap)
6. [ ] Interpret results and identify candidates
```

#### Literature Review
```markdown
1. [ ] Define search terms and databases
2. [ ] Retrieve relevant papers (tool: search_pubmed)
3. [ ] Extract key findings
4. [ ] Synthesize evidence
5. [ ] Identify gaps and conflicts
```

### Anti-Patterns to Avoid
❌ **Don't**: Create monolithic "analyze all data" steps
✅ **Do**: Break into specific, testable sub-steps

❌ **Don't**: Assume tools will work without validation
✅ **Do**: Include verification steps after tool usage

❌ **Don't**: Plan without checking resource availability
✅ **Do**: Cross-reference tools/data against registries

❌ **Don't**: Ignore budget constraints
✅ **Do**: Provide fallback options if budget is tight

---

## Interaction with Other Agents

### Handoff to LEARN
PLAN provides:
- List of required knowledge gaps
- Data sources to query
- Papers/references to retrieve
- Specific questions to answer

LEARN uses plan to:
- Focus retrieval on relevant topics
- Validate assumptions
- Fill knowledge gaps
- Refine step estimates

### Handoff to EXECUTE
PLAN provides:
- Ordered checklist of executable steps
- Tool names and parameters
- Data file paths
- Success criteria for validation

EXECUTE uses plan to:
- Sequence operations correctly
- Select appropriate tools
- Validate intermediate outputs
- Report progress per step

### Feedback from ASSESS
ASSESS can trigger plan revision if:
- Original approach is flawed
- Resources are insufficient
- Assumptions were incorrect
- Better approach discovered

When revising:
1. Keep working steps marked [✓]
2. Mark failed steps [✗] with reason
3. Insert corrected or new steps
4. Update resource estimates

---

## Self-Evaluation Criteria (AGE Framework)

The PLAN agent evaluates itself on:

### Goal Achievement (0-10)
- Did the plan address the user's query?
- Are all requirements covered?

### Efficiency (0-10)
- Is the plan optimal or over-complicated?
- Are there redundant steps?

### Quality (0-10)
- Is the plan clear and executable?
- Are risks and assumptions documented?

### Clarity (0-10)
- Can a downstream agent execute without ambiguity?
- Are success criteria measurable?

### Self-Reflection Questions
1. "Did I identify all critical resources?"
2. "Are my time/cost estimates realistic?"
3. "Did I consider failure scenarios?"
4. "Is there a simpler approach I missed?"
5. "Are my assumptions documented and reasonable?"

---

## Version History

### v2.0 (2025-10-30)
- Added budget-aware planning
- Enhanced risk assessment framework
- Added self-evaluation (AGE) criteria
- Improved step dependency tracking
- Added version management support

### v1.5 (2025-09-15)
- Added custom tool/data support
- Improved retrieval integration
- Enhanced quality scoring rubrics

### v1.0 (2025-09-01)
- Initial implementation
- Basic task decomposition
- Tool/data identification
- Checklist format

---

## Examples

### Example 1: CRISPR Screen Design

**Input:**
```python
{
    "user_query": "Design a pooled CRISPR screen to identify genes regulating T cell exhaustion. Generate 32 target genes.",
    "budget": {"time_seconds": 600, "cost_usd": 3.0}
}
```

**Output:**
```markdown
## Plan for: CRISPR Screen for T Cell Exhaustion Genes

### High-Level Strategy
We will identify candidate genes from literature and databases, design a focused 
sgRNA library targeting 32 genes, calculate experimental parameters, and output 
a validated library with protocol.

### Steps
1. [ ] Define biological context and hypothesis
   - Tools: None (reasoning step)
   - Success: Clear statement of exhaustion phenotype and expected regulators
   - Est. Time: 30s, Cost: $0.05

2. [ ] Search literature for T cell exhaustion genes
   - Tools: search_pubmed, extract_gene_mentions
   - Data: PubMed abstracts
   - Success: List of 50+ candidate genes with references
   - Est. Time: 60s, Cost: $0.20

3. [ ] Query pathway databases for exhaustion-related genes
   - Tools: query_kegg_pathway, query_reactome
   - Data: KEGG, Reactome databases
   - Success: Pathway-annotated gene list
   - Est. Time: 45s, Cost: $0.10

4. [ ] Rank and select top 32 target genes
   - Tools: rank_genes_by_literature_support
   - Success: 32 genes with justification for each
   - Est. Time: 60s, Cost: $0.15

5. [ ] Design sgRNA library for selected genes
   - Tools: design_crispr_library
   - Data: Human genome reference
   - Success: 4-6 sgRNAs per gene, validated for specificity
   - Est. Time: 120s, Cost: $0.40

6. [ ] Calculate experimental parameters
   - Tools: calculate_screen_coverage, calculate_moi
   - Success: Cell numbers, MOI, timepoints specified
   - Est. Time: 30s, Cost: $0.05

7. [ ] Generate validation metrics
   - Tools: calculate_on_target_scores, predict_off_targets
   - Success: Quality metrics for each sgRNA
   - Est. Time: 60s, Cost: $0.20

8. [ ] Output library and protocol
   - Tools: generate_library_csv, format_protocol
   - Success: CSV file with library, text protocol
   - Est. Time: 45s, Cost: $0.10

### Assumptions
- Human T cells are the model system
- Pooled screen format (not arrayed)
- Standard Cas9 system (not base editors)
- 4-6 sgRNAs per gene is sufficient

### Risks
- **Medium**: Literature search may yield <32 high-confidence genes
  → Mitigation: Include pathway-based candidates to reach 32
- **Low**: sgRNA design may fail for some genes (e.g., repetitive regions)
  → Mitigation: Design 8 sgRNAs per gene, select best 4-6

### Resource Requirements
- Total Time: 450s (75% of budget)
- Total Cost: $1.25 (42% of budget)
- Critical Tools: design_crispr_library, search_pubmed
- Critical Data: PubMed, human genome reference
```

---

## Appendix: Integration with PM Agent

The PM Agent monitors PLAN phase and collects:

**Metrics:**
- Time taken for planning
- Tokens used
- Number of steps generated
- Estimated vs actual budget usage

**Quality Assessment:**
- Clarity score (manual or LLM-based)
- Completeness (requirements coverage)
- Feasibility (resource availability check)

**Feedback Loop:**
If PLAN quality is low (<0.6), PM Agent can:
1. Request re-planning with clearer instructions
2. Suggest simplification
3. Recommend different approach
4. Flag for human review

---

*Agent Specification: PLAN*
*Version: 2.0*
*Last Updated: 2025-10-30*
*Owner: BioPLE Development Team*
