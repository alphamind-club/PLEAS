import { readFile, writeFile } from 'fs/promises'

import { appendJournalEvent, readEnabledTools, writeEnabledTools } from './journal.js'
import type { BioPleaseProject, BioPleaseToolRecord } from './types.js'

type RegistryServer = {
  id: string
  name: string
  description: string
  url: string
  tags: string[]
}

const FALLBACK_REGISTRY: RegistryServer[] = [
  {
    id: 'biocontext_kb',
    name: 'BioContextAI Knowledgebase MCP',
    description:
      'Broad biomedical knowledgebase covering STRING, Open Targets, Reactome, UniProt, Human Protein Atlas, EuropePMC, AlphaFold, Ensembl, PRIDE, ClinicalTrials.gov, and more.',
    url: 'https://biocontext-kb.fastmcp.app/mcp',
    tags: ['biocontext', 'knowledgebase', 'literature', 'pathways', 'targets'],
  },
  {
    id: 'ebi_ols',
    name: 'EMBL-EBI Ontology Lookup Service (OLS)',
    description:
      'Biomedical ontology lookups and semantic grounding for disease, assay, and phenotype terms.',
    url: 'https://www.ebi.ac.uk/ols4/api/mcp',
    tags: ['ontology', 'semantics', 'disease', 'phenotype'],
  },
  {
    id: 'open_targets',
    name: 'Open Targets Platform MCP',
    description:
      'Target-disease associations, variants, drugs, GWAS, and therapeutic prioritisation data.',
    url: 'https://mcp.platform.opentargets.org/mcp',
    tags: ['drug', 'disease', 'targets', 'gwas', 'variants'],
  },
  {
    id: 'clinicaltrials',
    name: 'ClinicalTrials.gov MCP Server',
    description:
      'Search, retrieve, and analyze clinical study data programmatically.',
    url: 'https://clinicaltrialsgov-mcp-server.fastmcp.app/mcp',
    tags: ['clinical trials', 'studies', 'patients'],
  },
  {
    id: 'nucleotide_archive',
    name: 'Nucleotide Archive MCP server',
    description:
      'Find and access RNA sequencing datasets from ENA for validation and reanalysis.',
    url: 'https://nucleotide-archive-mcp.fastmcp.app/mcp',
    tags: ['rna-seq', 'single-cell', 'datasets', 'ena'],
  },
  {
    id: 'string_mcp',
    name: 'STRING MCP Server',
    description:
      'Protein interaction networks and STRING database access.',
    url: 'https://mcp.string-db.org/',
    tags: ['stringdb', 'ppi', 'interaction networks'],
  },
]

function sanitizeServerId(input: string): string {
  return input.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
}

function scoreServer(server: RegistryServer, goal: string): number {
  const haystack = `${server.name} ${server.description} ${server.tags.join(' ')}`.toLowerCase()
  return goal
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .filter(token => token.length > 2)
    .reduce((score, token) => (haystack.includes(token) ? score + 1 : score), 0)
}

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try {
    return JSON.parse(await readFile(path, 'utf8')) as T
  } catch {
    return fallback
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, 'utf8')
}

function toToolRecord(server: RegistryServer, provenance: string): BioPleaseToolRecord {
  return {
    id: server.id,
    name: server.name,
    description: server.description,
    source: 'biocontext',
    enabledAt: new Date().toISOString(),
    remote: true,
    url: server.url,
    tags: server.tags,
    provenance,
  }
}

function mergeRegistryServers(servers: RegistryServer[]): RegistryServer[] {
  const map = new Map<string, RegistryServer>()
  for (const server of [...FALLBACK_REGISTRY, ...servers]) {
    map.set(server.id, server)
  }
  return [...map.values()]
}

function parseRegistryHtml(html: string): RegistryServer[] {
  const servers: RegistryServer[] = []
  const cardPattern =
    /itemType="https:\/\/schema\.org\/SoftwareApplication"[\s\S]*?itemProp="name"><a[^>]*href="[^"]+">([^<]+)<\/a>[\s\S]*?itemProp="description">([\s\S]*?)<\/div>[\s\S]*?(https:\/\/[^"<\s]+(?:\/mcp|\/))/g

  for (const match of html.matchAll(cardPattern)) {
    const name = match[1]?.replace(/\s+/g, ' ').trim()
    const description = match[2]
      ?.replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
    const url = match[3]?.trim()
    if (!name || !description || !url) {
      continue
    }
    servers.push({
      id: sanitizeServerId(name),
      name,
      description,
      url,
      tags: [],
    })
  }

  return servers
}

export async function fetchRegistryServers(
  project: BioPleaseProject,
  force = false,
): Promise<RegistryServer[]> {
  const cachePath = `${project.paths.cache}/registry-cache.json`
  const cached = await readJson<{ generatedAt: string | null; servers: RegistryServer[] }>(
    cachePath,
    { generatedAt: null, servers: FALLBACK_REGISTRY },
  )

  const now = Date.now()
  const cacheAge =
    cached.generatedAt === null ? Number.POSITIVE_INFINITY : now - Date.parse(cached.generatedAt)

  if (!force && cacheAge <= project.config.bioContext.cacheTtlMs && cached.servers.length > 0) {
    return mergeRegistryServers(cached.servers)
  }

  try {
    const response = await fetch(project.config.bioContext.registryUrl)
    if (!response.ok) {
      throw new Error(`Registry request failed with ${response.status}`)
    }
    const html = await response.text()
    const parsed = parseRegistryHtml(html)
    const merged = mergeRegistryServers(parsed)
    await writeJson(cachePath, {
      generatedAt: new Date().toISOString(),
      servers: merged,
    })
    return merged
  } catch {
    await writeJson(cachePath, {
      generatedAt: new Date().toISOString(),
      servers: FALLBACK_REGISTRY,
    })
    return FALLBACK_REGISTRY
  }
}

async function upsertMcpServers(
  project: BioPleaseProject,
  records: RegistryServer[],
): Promise<void> {
  const current = await readJson<{ mcpServers: Record<string, unknown> }>(
    project.paths.mcpConfig,
    { mcpServers: {} },
  )

  for (const record of records) {
    current.mcpServers[record.id] = {
      type: 'http',
      url: record.url,
    }
  }

  await writeJson(project.paths.mcpConfig, current)
}

export async function ensureKnowledgebaseTool(
  project: BioPleaseProject,
): Promise<BioPleaseToolRecord[]> {
  const enabled = await readEnabledTools(project)
  if (enabled.some(tool => tool.id === 'biocontext_kb')) {
    return enabled
  }

  const kb = FALLBACK_REGISTRY[0]!
  const next = [...enabled, toToolRecord(kb, 'default')]
  await upsertMcpServers(project, [kb])
  await writeEnabledTools(project, next)
  await appendJournalEvent(
    project,
    'tool_enabled',
    {
      toolId: kb.id,
      toolName: kb.name,
      url: kb.url,
      provenance: 'default',
    },
  )
  return next
}

export async function autoInstallRelevantTools(
  project: BioPleaseProject,
  goal: string,
): Promise<BioPleaseToolRecord[]> {
  let enabled = await ensureKnowledgebaseTool(project)
  if (!project.config.bioContext.autoInstall) {
    return enabled
  }

  const registry = await fetchRegistryServers(project)
  const enabledIds = new Set(enabled.map(tool => tool.id))
  const additions = registry
    .filter(server => !enabledIds.has(server.id))
    .map(server => ({ server, score: scoreServer(server, goal) }))
    .filter(item => item.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, 3)
    .map(item => item.server)

  if (additions.length === 0) {
    return enabled
  }

  await upsertMcpServers(project, additions)

  const addedRecords = additions.map(server => toToolRecord(server, 'auto-install'))
  enabled = [...enabled, ...addedRecords]
  await writeEnabledTools(project, enabled)

  for (const tool of addedRecords) {
    await appendJournalEvent(project, 'tool_enabled', {
      toolId: tool.id,
      toolName: tool.name,
      url: tool.url ?? null,
      provenance: tool.provenance ?? 'auto-install',
      goal,
    })
  }

  return enabled
}

export async function revokeTool(
  project: BioPleaseProject,
  toolId: string,
): Promise<BioPleaseToolRecord[]> {
  const enabled = await readEnabledTools(project)
  const next = enabled.filter(tool => tool.id !== toolId)
  if (next.length === enabled.length) {
    return enabled
  }

  const current = await readJson<{ mcpServers: Record<string, unknown> }>(
    project.paths.mcpConfig,
    { mcpServers: {} },
  )
  delete current.mcpServers[toolId]
  await writeJson(project.paths.mcpConfig, current)
  await writeEnabledTools(project, next)
  await appendJournalEvent(project, 'tool_revoked', { toolId })
  return next
}
