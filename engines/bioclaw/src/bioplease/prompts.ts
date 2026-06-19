import { basename, relative } from 'path'

import type {
  BioPleaseConfig,
  BioPleasePhase,
  BioPleaseProgressItem,
  BioPleaseProject,
  BioPleaseToolRecord,
  WorkspaceLedger,
} from './types.js'

export type BioPleaseSkillBundle = {
  dataLakeFiles: string[]
  toolModules: string[]
  toolDescriptions: string[]
  envScripts: string[]
}

const FRAMEWORK_OVERVIEW = `You are BioPLEASE running inside OpenClaude in a local-first coding environment.

Product mode:
- PLAN: maintain an explicit living plan and task checklist
- LEARN: use BioContextAI and bundled biomedical resources before guessing
- EXECUTE: write, run, and fix code directly inside the opened working folder
- ASSESS: review evidence, failure modes, reproducibility, and scientific quality
- SHARE: leave behind summaries, reports, artifacts, and a clear next-step state

Core rules:
1. Stay anchored to the opened folder. Do not scatter work into random temp locations.
2. Treat .bioplease/state.json, .bioplease/plan.md, .bioplease/journal.ndjson, and .bioplease/artifact-manifest.json as the runtime source of truth.
3. Put reusable instructions in .claude/skills and keep project facts in memory files, not in vague chat-only recollections.
4. Prefer extending or repairing the existing code path over starting over.
5. Update the plan, summary, and task state when reality changes.`

export function buildProjectClaudeMd(params: {
  config: BioPleaseConfig
  projectRoot: string
  repoRoot: string
}): string {
  const pubmedCommand = relative(
    params.projectRoot,
    `${params.repoRoot}/scripts/bioplease/pubmed.py`,
  ).replaceAll('\\', '/')
  const europePmcCommand = relative(
    params.projectRoot,
    `${params.repoRoot}/scripts/bioplease/europe_pmc.py`,
  ).replaceAll('\\', '/')

  return `# BioPLEASE Codex Contract

${FRAMEWORK_OVERVIEW}

Project:
- Title: ${params.config.title}
- Research question: ${params.config.researchQuestion}
- Background context: ${params.config.backgroundContext || 'None recorded yet.'}
- Framework version: ${params.config.frameworkVersion}

Runtime structure:
- The opened folder is the main working directory for code and outputs.
- Hidden runtime metadata lives under \`.bioplease/\`.
- Stable reusable skills live under \`.claude/skills/\`.
- Reports should land in \`reports/\`, figures in \`figures/\`, data outputs in \`data/\`, and packaged deliverables in \`artifacts/\`.

Planning rules:
1. Keep \`.bioplease/plan.md\` current and concrete.
2. The plan may change, but it must stay truthful.
3. Maintain unfinished tasks instead of forgetting them between turns.

Learning rules:
1. Use BioContextAI Knowledgebase MCP as the default biomedical retrieval layer.
2. Auto-discovered MCP servers from the BioContext registry can be enabled when they materially help the current task.
3. For direct literature search, you may use:
   - \`python ${pubmedCommand} search --query "<query>" --limit 8\`
   - \`python ${europePmcCommand} search --query "<query>" --limit 8\`

Execution rules:
1. Work directly inside this opened folder.
2. Leave code, generated apps, analyses, and outputs in this folder unless a tool explicitly requires another location.
3. Track key file locations in memory and plan updates so future turns can find them reliably.

Assessment and sharing rules:
1. Keep \`.bioplease/summaries/latest.md\` concise and current.
2. Put durable writeups in \`reports/\`.
3. If blocked, record the blocker in \`.bioplease/state.json\` and the plan instead of looping.
`
}

export function buildMemoryEntrypoint(): string {
  return `# MEMORY

- Record durable project facts, file locations, naming conventions, environment notes, active preview URLs, and important failures.
- Prefer short bullet links to more detailed topic files in this directory.
- Do not use memory for generic chat summaries.`
}

export function buildPlanTemplate(config: BioPleaseConfig): string {
  return `# BioPLEASE Plan

Project: ${config.title}
Research question: ${config.researchQuestion}

## Mission
- Deliver working biomedical code and outputs directly in this folder.
- Keep the project recoverable across retries and resumed sessions.

## Current plan
- [ ] Reconfirm the goal and constraints
- [ ] Gather biomedical context and tools for the active task
- [ ] Implement or repair the primary code path
- [ ] Review outputs, risks, and reproducibility
- [ ] Update the latest summary and next steps
`
}

export function buildCurrentTaskTemplate(goal = 'No active goal recorded yet.'): string {
  return `# Current Task

Goal: ${goal}

Notes:
- Update this file when the active objective materially changes.
- Point to important files or blockers here if they would be easy to lose track of.
`
}

export function buildLatestSummaryTemplate(config: BioPleaseConfig): string {
  return `# Latest Summary

Project: ${config.title}

## Status
- No run summary has been recorded yet.

## Most recent outcome
- Pending.

## Next best action
- Clarify the next execution goal and start the BioPLEASE workflow.
`
}

export function buildSessionSummaryMarkdown(params: {
  sessionId: string
  goal: string
  latestSummary: string
  createdFiles: string[]
  modifiedFiles: string[]
  deletedFiles: string[]
}): string {
  const renderSection = (title: string, values: string[]) =>
    values.length === 0
      ? `## ${title}\n- None\n`
      : `## ${title}\n${values.map(value => `- ${value}`).join('\n')}\n`

  return `# Session ${params.sessionId}

## Goal
${params.goal}

## Summary
${params.latestSummary || 'No summary was produced.'}

${renderSection('Created Files', params.createdFiles)}
${renderSection('Modified Files', params.modifiedFiles)}
${renderSection('Deleted Files', params.deletedFiles)}`
}

export function renderWorkspaceLedgerMarkdown(ledger: WorkspaceLedger): string {
  const renderList = (label: string, values: string[]) =>
    `- ${label}: ${values.length > 0 ? values.join(', ') : 'none'}`

  return [
    `- Generated at: ${ledger.generatedAt}`,
    `- Current working directory: ${ledger.currentWorkingDirectory}`,
    `- Active task summary: ${ledger.activeTaskSummary ?? 'none'}`,
    renderList('Recent files', ledger.recentFiles),
    renderList('Primary scripts', ledger.primaryScripts),
    renderList('Datasets', ledger.datasets),
    renderList('Figures', ledger.figures),
    renderList('Outputs', ledger.outputs),
    renderList('Unfinished tasks', ledger.unfinishedTasks),
    renderList('Failed commands', ledger.failedCommands),
  ].join('\n')
}

function renderProgress(progressItems: BioPleaseProgressItem[]): string {
  if (progressItems.length === 0) {
    return '- No progress items recorded yet.'
  }

  return progressItems
    .map(item => {
      const status =
        item.status === 'done'
          ? 'done'
          : item.status === 'in_progress'
            ? 'in progress'
            : item.status
      const suffix = item.detail ? ` :: ${item.detail}` : ''
      return `- [${status}] ${item.phase} :: ${item.title}${suffix}`
    })
    .join('\n')
}

function renderEnabledTools(tools: BioPleaseToolRecord[]): string {
  if (tools.length === 0) {
    return '- No extra BioContext or BioPLEASE tools are enabled yet.'
  }

  return tools
    .slice(0, 12)
    .map(tool => {
      const source = tool.remote ? 'remote' : 'local'
      return `- ${tool.name} (${tool.source}, ${source})${tool.url ? ` :: ${tool.url}` : ''}`
    })
    .join('\n')
}

function buildPhaseMission(phase: BioPleasePhase): string {
  switch (phase) {
    case 'Plan':
      return 'Refresh .bioplease/plan.md into a concrete, truthful task plan and update progress state.'
    case 'Learn':
      return 'Ground the work with BioContext, bundled biomedical skills, literature, datasets, and tool selection.'
    case 'Execute':
      return 'Write, run, debug, and refine code directly in the opened folder, keeping outputs organized and discoverable.'
    case 'Assess':
      return 'Review scientific quality, correctness, blockers, reproducibility, and whether the outputs answer the goal.'
    case 'Share':
      return 'Produce the most recent summary, report pointers, and clear next steps for the user.'
  }
}

function buildPhaseGuardrails(phase: BioPleasePhase): string {
  switch (phase) {
    case 'Plan':
      return `Phase-specific rules:
1. Stay in planning mode. Read project files, update \`.bioplease/plan.md\`, \`.bioplease/current-task.md\`, and related runtime state.
2. Read the smallest set of files needed. Start with \`.bioplease/plan.md\` and \`.bioplease/current-task.md\`, then consult state, journal, or memory only if they are needed to resolve ambiguity.
2. Do not create or modify product code, datasets, reports, or final outputs in this phase.
3. Do not run implementation commands, test commands, build commands, or code-generation Bash flows in this phase.
4. Only update memory if you discovered a new durable project fact that future turns would otherwise lose.
5. If the plan depends on execution, record that as the next Execute task instead of doing it now.`
    case 'Learn':
      return `Phase-specific rules:
1. Focus on retrieval, grounding, tool discovery, and environment awareness.
2. Prefer BioContextAI, bundled BioPLEASE skills, and literature or registry lookups before guessing.
3. If the task is a simple local coding task and no new biomedical context is required, keep this phase short, record that the current toolset is sufficient, and hand off to Execute.
4. Prefer one or two targeted retrieval or tool-discovery actions over broad catalog exploration.
5. Do not perform the main code implementation in this phase.
6. If a new tool or dataset is important, record why it matters and where it came from.`
    case 'Execute':
      return `Phase-specific rules:
1. This is the implementation phase. Write, run, debug, and refine code directly in the opened folder.
2. Keep generated outputs in stable, rediscoverable locations and record them in plan, memory, or summary updates.
3. Prefer small verifiable changes over broad rewrites.
4. If blocked, leave concrete evidence of the failure and the most useful next step.`
    case 'Assess':
      return `Phase-specific rules:
1. Review correctness, reproducibility, risk, and scientific quality of the current outputs.
2. Prefer validation, inspection, and focused checks over new feature work.
3. Only make the smallest fixes needed to address clear defects discovered during assessment.
4. Record unresolved risks and blockers explicitly if they remain.`
    case 'Share':
      return `Phase-specific rules:
1. Update summaries, reports, and next steps so another session can resume cleanly.
2. Do not start new implementation work in this phase.
3. Point to important files, outputs, and remaining blockers explicitly.
4. Leave the user with a truthful snapshot of what was completed and what still needs work.`
  }
}

export function buildPhaseAppendPrompt(params: {
  project: BioPleaseProject
  phase: BioPleasePhase
  goal: string
  ledger: WorkspaceLedger
  progressItems: BioPleaseProgressItem[]
  enabledTools: BioPleaseToolRecord[]
  currentSummary: string
}): string {
  return `${FRAMEWORK_OVERVIEW}

Current phase: ${params.phase}
Phase mission: ${buildPhaseMission(params.phase)}
Current goal: ${params.goal}

Project root:
\`${params.project.root}\`

BioPLEASE runtime files:
- \`.bioplease/plan.md\`
- \`.bioplease/state.json\`
- \`.bioplease/current-task.md\`
- \`.bioplease/journal.ndjson\`
- \`.bioplease/memory/MEMORY.md\`

Workspace ledger:
${renderWorkspaceLedgerMarkdown(params.ledger)}

Progress checklist:
${renderProgress(params.progressItems)}

Enabled BioContext and BioPLEASE tools:
${renderEnabledTools(params.enabledTools)}

Most recent summary:
${params.currentSummary || 'No summary recorded yet.'}

Non-negotiable operating rules:
1. Stay inside the opened folder and keep important outputs in stable locations.
2. Update \`.bioplease/plan.md\` if the plan is stale, incomplete, or contradicted by reality.
3. If you create important files, make them easy to rediscover by recording them in plan/memory/summary.
4. Avoid thrashing. If blocked, state the blocker, leave evidence, and move to the most productive next action.
5. When relevant, use the generated BioPLEASE skills in \`.claude/skills/\` before improvising.

${buildPhaseGuardrails(params.phase)}
`
}

export function buildPhaseUserPrompt(params: {
  config: BioPleaseConfig
  phase: BioPleasePhase
  goal: string
}): string {
  return `Project: ${params.config.title}
Research question: ${params.config.researchQuestion}

Background context:
${params.config.backgroundContext || 'None recorded yet.'}

Current phase: ${params.phase}
Current goal:
${params.goal}

Do the work for this phase autonomously, follow the phase-specific rules, update the project-local BioPLEASE files as needed, and leave the folder in a more recoverable state than you found it.

${buildPhaseGuardrails(params.phase)}`
}

export function buildDataLakeSkill(params: {
  repoRoot: string
  dataLakeFiles: string[]
}): string {
  const preview = params.dataLakeFiles.slice(0, 18).map(file => `- ${file}`).join('\n')
  return `---
description: BioPLEASE data lake catalog and usage guidance
when_to_use: Use when the task could benefit from searching or reusing the bundled BioPLEASE data lake instead of downloading a new dataset from scratch.
---

# BioPLEASE Data Lake

The bundled BioPLEASE data lake lives at:
\`${params.repoRoot.replaceAll('\\', '/')}/BioPlease_tools/data_lake\`

Use it as a read-only catalog unless the user explicitly asks to modify those assets.

Representative files:
${preview || '- No files were indexed.'}

Guidance:
- Search this catalog before downloading redundant biomedical datasets.
- Copy only the specific files needed into the opened working folder when a run needs local writable outputs.
- Record provenance in reports or summaries when using a bundled dataset.
`
}

export function buildEnvironmentSkill(params: {
  repoRoot: string
  envScripts: string[]
}): string {
  const scripts = params.envScripts.map(script => `- ${script}`).join('\n')
  return `---
description: BioPLEASE environment bootstrap and diagnostics guidance
when_to_use: Use when the task needs biomedical software bootstrap, environment diagnosis, or awareness of the bundled BioPLEASE environment assets.
---

# BioPLEASE Environment

The environment bundle lives at:
\`${params.repoRoot.replaceAll('\\', '/')}/BioPlease_tools/bioplease_env\`

Important files:
${scripts || '- No environment scripts were indexed.'}

Rules:
- Treat these assets as explicit bootstrap helpers, not silent background mutations.
- Explain when an environment change is necessary.
- Prefer diagnostics first, then the smallest environment adjustment that unblocks progress.
`
}

export function buildToolCatalogSkill(params: {
  repoRoot: string
  toolModules: string[]
  toolDescriptions: string[]
}): string {
  const modules = params.toolModules.slice(0, 24).map(name => `- ${name}`).join('\n')
  const descriptions = params.toolDescriptions
    .slice(0, 24)
    .map(name => `- ${name}`)
    .join('\n')

  return `---
description: BioPLEASE biomedical tool catalog imported from the bundled BioPLEASE tool universe
when_to_use: Use when the task needs domain-specific biomedical operations, schema lookups, or inspiration from the imported BioPLEASE tool ecosystem.
---

# BioPLEASE Tool Catalog

Bundled source root:
\`${params.repoRoot.replaceAll('\\', '/')}/BioPlease_tools/tool\`

Tool modules:
${modules || '- No tool modules were indexed.'}

Tool description modules:
${descriptions || '- No description modules were indexed.'}

Guidance:
- Review the imported catalog before inventing a new biomedical tool flow.
- Reuse naming and task structure from the bundle when it fits the current goal.
- If a missing capability would help, use LEARN to look for a BioContextAI MCP server first.
`
}

export function buildLearnSkill(params: {
  knowledgebaseUrl: string
  registryUrl: string
}): string {
  return `---
description: BioPLEASE LEARN workflow using BioContextAI
when_to_use: Use when a task needs biomedical literature, databases, ontologies, or a new MCP tool from the BioContextAI ecosystem.
---

# BioPLEASE LEARN

Default Knowledgebase MCP:
\`${params.knowledgebaseUrl}\`

Registry source:
\`${params.registryUrl}\`

Workflow:
1. Start with the Knowledgebase MCP for biomedical retrieval and grounding.
2. If the task still lacks a capability, inspect the BioContext registry and enable a compatible MCP server.
3. Record newly enabled tools in the BioPLEASE state and journal.
4. Prefer the smallest set of tools that solves the task cleanly.
`
}

export function buildSkillBundleIndex(
  projectRoot: string,
  bundle: BioPleaseSkillBundle,
): string {
  return `# BioPLEASE Skill Index

This project contains generated BioPLEASE skills in:
\`${projectRoot.replaceAll('\\', '/')}/.claude/skills\`

Skill coverage:
- Data lake files indexed: ${bundle.dataLakeFiles.length}
- Tool modules indexed: ${bundle.toolModules.length}
- Tool descriptions indexed: ${bundle.toolDescriptions.length}
- Environment scripts indexed: ${bundle.envScripts.length}
`
}

export function buildProjectDisplayName(projectRoot: string): string {
  return basename(projectRoot)
}
