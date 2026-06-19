import { mkdir, readFile, readdir, stat, writeFile } from 'fs/promises'
import { basename, join, relative, resolve } from 'path'
import { fileURLToPath } from 'url'

import {
  BIOPLEASE_FRAMEWORK_VERSION,
  type BioPleaseConfig,
  type BioPleasePaths,
  type BioPleasePhaseModelRouting,
  type BioPleaseProject,
  type BioPleaseProviderProfile,
  type BioPleaseRuntimeState,
  type WorkspaceLedger,
} from './types.js'
import {
  type BioPleaseSkillBundle,
  buildCurrentTaskTemplate,
  buildDataLakeSkill,
  buildEnvironmentSkill,
  buildLatestSummaryTemplate,
  buildLearnSkill,
  buildMemoryEntrypoint,
  buildPlanTemplate,
  buildProjectClaudeMd,
  buildSkillBundleIndex,
  buildToolCatalogSkill,
} from './prompts.js'

export const PROJECT_DIRECTORIES = [
  '.bioplease',
  '.bioplease/cache',
  '.bioplease/tooling',
  '.bioplease/summaries',
  '.bioplease/transcripts',
  '.bioplease/memory',
  '.bioplease/plans',
  '.claude',
  '.claude/skills',
  'reports',
  'artifacts',
  'figures',
  'data',
] as const

const DEFAULT_BIOCONTEXT_REGISTRY_URL = 'https://biocontext.ai/registry'
const DEFAULT_BIOCONTEXT_KB_URL = 'https://biocontext-kb.fastmcp.app/mcp'

export function resolveRepoRoot(): string {
  return resolve(fileURLToPath(new URL('../../', import.meta.url)))
}

export function normalizeProjectSlug(input: string): string {
  const cleaned = input
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return cleaned || 'bioplease-project'
}

export function getProjectPaths(projectRoot: string): BioPleasePaths {
  const root = resolve(projectRoot)
  const metadataRoot = join(root, '.bioplease')
  return {
    metadataRoot,
    cache: join(metadataRoot, 'cache'),
    tooling: join(metadataRoot, 'tooling'),
    summaries: join(metadataRoot, 'summaries'),
    transcripts: join(metadataRoot, 'transcripts'),
    memory: join(metadataRoot, 'memory'),
    plans: join(metadataRoot, 'plans'),
    config: join(metadataRoot, 'project.json'),
    state: join(metadataRoot, 'state.json'),
    plan: join(metadataRoot, 'plan.md'),
    planHistory: join(metadataRoot, 'plan-history.ndjson'),
    journal: join(metadataRoot, 'journal.ndjson'),
    artifactManifest: join(metadataRoot, 'artifact-manifest.json'),
    currentTask: join(metadataRoot, 'current-task.md'),
    reports: join(root, 'reports'),
    artifacts: join(root, 'artifacts'),
    figures: join(root, 'figures'),
    data: join(root, 'data'),
    settingsLocal: join(root, '.claude', 'settings.local.json'),
    skillsRoot: join(root, '.claude', 'skills'),
    projectClaude: join(root, 'CLAUDE.md'),
    mcpConfig: join(root, '.mcp.json'),
    sessionSummary: join(metadataRoot, 'summaries', 'latest.md'),
  }
}

export function getDefaultPhaseModelRouting(): BioPleasePhaseModelRouting {
  return {
    Plan: 'gpt-5.4-mini',
    Learn: 'gpt-5.4-mini',
    Execute: 'minimax-m2.7',
    Assess: 'gpt-5.4-mini',
    Share: 'gpt-5.4-mini',
    fallback: 'gpt-5.4-mini',
  }
}

export function resolveProviderProfiles(): Record<string, BioPleaseProviderProfile> {
  return {
    'gpt-5.4-mini': {
      model: 'gpt-5.4-mini',
      baseUrl: 'https://api.openai.com/v1',
      apiKey: process.env.OPENAI_API_KEY ?? '',
    },
    'minimax-m2.7': {
      model: 'minimax-m2.7',
      baseUrl: 'https://api.minimax.io/v1',
      apiKey: process.env.MINIMAX_API_KEY ?? '',
    },
  }
}

function createDefaultConfig(params: {
  root: string
  title: string
  researchQuestion: string
  backgroundContext?: string
  toolCatalogRoots?: string[]
}): BioPleaseConfig {
  const now = new Date().toISOString()
  return {
    slug: normalizeProjectSlug(params.title || basename(params.root)),
    title: params.title || basename(params.root),
    researchQuestion: params.researchQuestion,
    backgroundContext: params.backgroundContext ?? '',
    frameworkVersion: BIOPLEASE_FRAMEWORK_VERSION,
    createdAt: now,
    updatedAt: now,
    localWorkspaceMode: 'folder-root',
    metadataDirectory: '.bioplease',
    phaseModelRouting: getDefaultPhaseModelRouting(),
    bioContext: {
      autoInstall: true,
      registryUrl: DEFAULT_BIOCONTEXT_REGISTRY_URL,
      cacheTtlMs: 1000 * 60 * 60 * 24,
      defaultKnowledgebaseUrl: DEFAULT_BIOCONTEXT_KB_URL,
    },
    toolCatalogRoots:
      params.toolCatalogRoots && params.toolCatalogRoots.length > 0
        ? params.toolCatalogRoots
        : [join(resolveRepoRoot(), 'BioPlease_tools')],
  }
}

function mergeConfig(
  existing: BioPleaseConfig | null,
  defaults: BioPleaseConfig,
  overrides: {
    title: string
    researchQuestion: string
    backgroundContext?: string
    toolCatalogRoots?: string[]
  },
): BioPleaseConfig {
  const now = new Date().toISOString()
  const routing = getDefaultPhaseModelRouting()
  return {
    ...(existing ?? defaults),
    slug: normalizeProjectSlug(existing?.slug ?? defaults.slug),
    title: overrides.title || existing?.title || defaults.title,
    researchQuestion:
      overrides.researchQuestion || existing?.researchQuestion || defaults.researchQuestion,
    backgroundContext:
      overrides.backgroundContext ?? existing?.backgroundContext ?? defaults.backgroundContext,
    frameworkVersion: BIOPLEASE_FRAMEWORK_VERSION,
    updatedAt: now,
    createdAt: existing?.createdAt ?? defaults.createdAt,
    localWorkspaceMode: 'folder-root',
    metadataDirectory: '.bioplease',
    phaseModelRouting: {
      ...(existing?.phaseModelRouting ?? {}),
      ...routing,
    },
    bioContext: {
      registryUrl:
        existing?.bioContext?.registryUrl ?? defaults.bioContext.registryUrl,
      cacheTtlMs:
        existing?.bioContext?.cacheTtlMs ?? defaults.bioContext.cacheTtlMs,
      defaultKnowledgebaseUrl:
        existing?.bioContext?.defaultKnowledgebaseUrl ??
        defaults.bioContext.defaultKnowledgebaseUrl,
      autoInstall:
        existing?.bioContext?.autoInstall ?? defaults.bioContext.autoInstall,
    },
    toolCatalogRoots:
      overrides.toolCatalogRoots && overrides.toolCatalogRoots.length > 0
        ? overrides.toolCatalogRoots
        : existing?.toolCatalogRoots?.length
          ? existing.toolCatalogRoots
          : defaults.toolCatalogRoots,
  }
}

async function readOptionalJson<T>(path: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(path, 'utf8')) as T
  } catch {
    return null
  }
}

async function writeIfMissing(path: string, content: string): Promise<void> {
  try {
    await stat(path)
  } catch {
    await writeFile(path, content, 'utf8')
  }
}

async function ensureIgnoredEntry(projectRoot: string): Promise<void> {
  const gitignorePath = join(projectRoot, '.gitignore')
  const required = ['.bioplease/', '.claude/settings.local.json']

  let current = ''
  try {
    current = await readFile(gitignorePath, 'utf8')
  } catch {
    current = ''
  }

  const next = current.trimEnd()
  const missing = required.filter(entry => !current.includes(entry))
  if (missing.length === 0) {
    return
  }

  const chunks = [next, ...missing].filter(Boolean)
  await writeFile(gitignorePath, `${chunks.join('\n')}\n`, 'utf8')
}

async function listRelativeFiles(
  root: string,
  options?: {
    include?: (relativePath: string, fullPath: string) => boolean
  },
): Promise<string[]> {
  const results: string[] = []

  async function walk(current: string): Promise<void> {
    let entries = await readdir(current, { withFileTypes: true })
    entries = entries.filter(entry => entry.name !== '__pycache__')
    for (const entry of entries) {
      const fullPath = join(current, entry.name)
      const relativePath = relative(root, fullPath).replaceAll('\\', '/')
      if (entry.isDirectory()) {
        await walk(fullPath)
        continue
      }
      if (!entry.isFile()) {
        continue
      }
      if (options?.include && !options.include(relativePath, fullPath)) {
        continue
      }
      results.push(relativePath)
    }
  }

  try {
    await walk(root)
  } catch {
    return []
  }

  results.sort((left, right) => left.localeCompare(right))
  return results
}

async function collectSkillBundle(repoRoot: string): Promise<BioPleaseSkillBundle> {
  const bundleRoot = join(repoRoot, 'BioPlease_tools')
  const dataLakeFiles = await listRelativeFiles(join(bundleRoot, 'data_lake'))
  const toolModules = await listRelativeFiles(join(bundleRoot, 'tool'), {
    include: relativePath =>
      relativePath.endsWith('.py') &&
      !relativePath.startsWith('tool_description/') &&
      !relativePath.startsWith('example_mcp_tools/') &&
      !relativePath.startsWith('schema_db/'),
  })
  const toolDescriptions = await listRelativeFiles(
    join(bundleRoot, 'tool', 'tool_description'),
    {
      include: relativePath => relativePath.endsWith('.py'),
    },
  )
  const envScripts = await listRelativeFiles(join(bundleRoot, 'bioplease_env'))

  return {
    dataLakeFiles,
    toolModules,
    toolDescriptions,
    envScripts,
  }
}

function buildSettingsLocal(
  config: BioPleaseConfig,
  existing: Record<string, unknown> | null,
): Record<string, unknown> {
  const profiles = resolveProviderProfiles()
  const generated = {
    plansDirectory: '.bioplease/plans',
    autoMemoryEnabled: true,
    autoMemoryDirectory: '.bioplease/memory',
    env: {
      BIOPLEASE_ACTIVE: '1',
    },
    agentModels: {
      'gpt-5.4-mini': {
        base_url: profiles['gpt-5.4-mini'].baseUrl,
        api_key: profiles['gpt-5.4-mini'].apiKey,
      },
      'minimax-m2.7': {
        base_url: profiles['minimax-m2.7'].baseUrl,
        api_key: profiles['minimax-m2.7'].apiKey,
      },
    },
    agentRouting: {
      ...(typeof existing?.agentRouting === 'object' && existing.agentRouting
        ? (existing.agentRouting as Record<string, string>)
        : {}),
      Plan: config.phaseModelRouting.Plan,
      Learn: config.phaseModelRouting.Learn,
      Execute: config.phaseModelRouting.Execute,
      Assess: config.phaseModelRouting.Assess,
      Share: config.phaseModelRouting.Share,
      default: config.phaseModelRouting.fallback,
    },
    bioplease: {
      ...(typeof existing?.bioplease === 'object' && existing.bioplease
        ? (existing.bioplease as Record<string, unknown>)
        : {}),
      localWorkspaceMode: config.localWorkspaceMode,
      metadataDirectory: config.metadataDirectory,
      phaseModelRouting: config.phaseModelRouting,
      bioContext: config.bioContext,
      toolCatalogRoots: config.toolCatalogRoots,
    },
  } satisfies Record<string, unknown>

  return {
    ...(existing ?? {}),
    ...generated,
    env: {
      ...(typeof existing?.env === 'object' && existing.env
        ? (existing.env as Record<string, string>)
        : {}),
      BIOPLEASE_ACTIVE: '1',
    },
    agentModels: {
      ...(typeof existing?.agentModels === 'object' && existing.agentModels
        ? (existing.agentModels as Record<string, unknown>)
        : {}),
      ...generated.agentModels,
    },
  }
}

function buildDefaultRuntimeState(project: BioPleaseProject): BioPleaseRuntimeState {
  return {
    version: project.config.frameworkVersion,
    projectRoot: project.root,
    updatedAt: new Date().toISOString(),
    activeGoal: null,
    activePhase: null,
    activeSessionId: null,
    latestSummary: 'No runs yet.',
    progressItems: [],
    phaseStates: [
      { phase: 'Plan', status: 'idle', model: project.config.phaseModelRouting.Plan },
      { phase: 'Learn', status: 'idle', model: project.config.phaseModelRouting.Learn },
      {
        phase: 'Execute',
        status: 'idle',
        model: project.config.phaseModelRouting.Execute,
      },
      {
        phase: 'Assess',
        status: 'idle',
        model: project.config.phaseModelRouting.Assess,
      },
      { phase: 'Share', status: 'idle', model: project.config.phaseModelRouting.Share },
    ],
    recentFiles: [],
    recentArtifacts: [],
    previewUrls: [],
    lastToolEvent: null,
    blockedReason: null,
    enabledTools: [],
    sessions: [],
  }
}

function buildDefaultLedger(project: BioPleaseProject): WorkspaceLedger {
  return {
    generatedAt: new Date().toISOString(),
    projectRoot: project.root,
    currentWorkingDirectory: project.root,
    activeTaskSummary: null,
    recentFiles: [],
    primaryScripts: [],
    datasets: [],
    figures: [],
    outputs: [],
    failedCommands: [],
    unfinishedTasks: [],
    artifactManifestPath: '.bioplease/artifact-manifest.json',
    reportStatus: {
      planExists: true,
      summaryExists: true,
      sessionSummaryExists: true,
    },
    fileCounts: {
      total: 0,
      hidden: 0,
      visible: 0,
      data: 0,
      figures: 0,
      reports: 0,
      artifacts: 0,
    },
  }
}

async function writeGeneratedSkills(
  project: BioPleaseProject,
  bundle: BioPleaseSkillBundle,
): Promise<void> {
  const entries: Array<{ name: string; content: string }> = [
    {
      name: 'bioplease-data-lake',
      content: buildDataLakeSkill({
        repoRoot: project.repoRoot,
        dataLakeFiles: bundle.dataLakeFiles,
      }),
    },
    {
      name: 'bioplease-environment',
      content: buildEnvironmentSkill({
        repoRoot: project.repoRoot,
        envScripts: bundle.envScripts,
      }),
    },
    {
      name: 'bioplease-tool-catalog',
      content: buildToolCatalogSkill({
        repoRoot: project.repoRoot,
        toolModules: bundle.toolModules,
        toolDescriptions: bundle.toolDescriptions,
      }),
    },
    {
      name: 'bioplease-learn',
      content: buildLearnSkill({
        knowledgebaseUrl: project.config.bioContext.defaultKnowledgebaseUrl,
        registryUrl: project.config.bioContext.registryUrl,
      }),
    },
    {
      name: 'bioplease-index',
      content: `---
description: Index of generated BioPLEASE project skills
when_to_use: Use when you need to discover the available BioPLEASE skills in this project.
---

${buildSkillBundleIndex(project.root, bundle)}
`,
    },
  ]

  for (const entry of entries) {
    const skillDir = join(project.paths.skillsRoot, entry.name)
    await mkdir(skillDir, { recursive: true })
    await writeFile(join(skillDir, 'SKILL.md'), entry.content, 'utf8')
  }
}

async function writeMcpConfig(
  project: BioPleaseProject,
  existing: Record<string, unknown> | null,
): Promise<void> {
  const mcpServers =
    typeof existing?.mcpServers === 'object' && existing.mcpServers
      ? { ...(existing.mcpServers as Record<string, unknown>) }
      : {}

  mcpServers.biocontext_kb = {
    type: 'http',
    url: project.config.bioContext.defaultKnowledgebaseUrl,
  }

  await writeFile(
    project.paths.mcpConfig,
    `${JSON.stringify({ mcpServers }, null, 2)}\n`,
    'utf8',
  )
}

export async function loadProject(projectRoot: string): Promise<BioPleaseProject> {
  const root = resolve(projectRoot)
  const repoRoot = resolveRepoRoot()
  const paths = getProjectPaths(root)
  const config = await readOptionalJson<BioPleaseConfig>(paths.config)
  if (!config) {
    throw new Error(`BioPLEASE project config not found in ${paths.config}`)
  }
  return { root, repoRoot, paths, config }
}

export async function ensureProjectWorkspace(params: {
  projectRoot: string
  title?: string
  researchQuestion?: string
  backgroundContext?: string
  toolCatalogRoots?: string[]
}): Promise<BioPleaseProject> {
  const root = resolve(params.projectRoot)
  const repoRoot = resolveRepoRoot()
  const paths = getProjectPaths(root)

  for (const directory of PROJECT_DIRECTORIES) {
    await mkdir(join(root, directory), { recursive: true })
  }

  const defaults = createDefaultConfig({
    root,
    title: params.title || basename(root),
    researchQuestion:
      params.researchQuestion || 'Research question not set yet.',
    backgroundContext: params.backgroundContext,
    toolCatalogRoots: params.toolCatalogRoots,
  })
  const existingConfig = await readOptionalJson<BioPleaseConfig>(paths.config)
  const config = mergeConfig(existingConfig, defaults, {
    title: params.title || existingConfig?.title || basename(root),
    researchQuestion:
      params.researchQuestion ||
      existingConfig?.researchQuestion ||
      'Research question not set yet.',
    backgroundContext: params.backgroundContext,
    toolCatalogRoots: params.toolCatalogRoots,
  })

  const project: BioPleaseProject = {
    root,
    repoRoot,
    paths,
    config,
  }

  await ensureIgnoredEntry(root)
  await writeFile(paths.config, `${JSON.stringify(config, null, 2)}\n`, 'utf8')

  const settingsExisting = await readOptionalJson<Record<string, unknown>>(
    paths.settingsLocal,
  )
  await writeFile(
    paths.settingsLocal,
    `${JSON.stringify(buildSettingsLocal(config, settingsExisting), null, 2)}\n`,
    'utf8',
  )

  const mcpExisting = await readOptionalJson<Record<string, unknown>>(paths.mcpConfig)
  await writeMcpConfig(project, mcpExisting)

  const bundle = await collectSkillBundle(repoRoot)
  await writeGeneratedSkills(project, bundle)

  await writeIfMissing(
    paths.projectClaude,
    buildProjectClaudeMd({
      config,
      projectRoot: root,
      repoRoot,
    }),
  )
  await writeIfMissing(join(paths.memory, 'MEMORY.md'), buildMemoryEntrypoint())
  await writeIfMissing(paths.plan, buildPlanTemplate(config))
  await writeIfMissing(paths.currentTask, buildCurrentTaskTemplate())
  await writeIfMissing(paths.sessionSummary, buildLatestSummaryTemplate(config))
  await writeIfMissing(paths.planHistory, '')
  await writeIfMissing(paths.journal, '')
  await writeIfMissing(paths.artifactManifest, '[]\n')
  await writeIfMissing(
    paths.state,
    `${JSON.stringify(buildDefaultRuntimeState(project), null, 2)}\n`,
  )
  await writeIfMissing(
    join(paths.cache, 'registry-cache.json'),
    `${JSON.stringify({ generatedAt: null, servers: [] }, null, 2)}\n`,
  )
  await writeIfMissing(
    join(paths.tooling, 'enabled-tools.json'),
    `${JSON.stringify(
      [
        {
          id: 'biocontext_kb',
          name: 'BioContextAI Knowledgebase MCP',
          description: 'Default biomedical retrieval MCP server',
          source: 'biocontext',
          enabledAt: new Date().toISOString(),
          remote: true,
          url: config.bioContext.defaultKnowledgebaseUrl,
          tags: ['biocontext', 'knowledgebase', 'mcp'],
          provenance: 'default',
        },
      ],
      null,
      2,
    )}\n`,
  )
  await writeIfMissing(
    join(paths.cache, 'workspace-ledger.json'),
    `${JSON.stringify(buildDefaultLedger(project), null, 2)}\n`,
  )

  return project
}
