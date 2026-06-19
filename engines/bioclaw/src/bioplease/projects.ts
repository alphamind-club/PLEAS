import { mkdir, readFile, writeFile } from 'fs/promises'
import { join, resolve } from 'path'

import { buildArtifactManifest } from './artifacts.js'
import { readEnabledTools, readJournal, readRuntimeState, readSessionSummary } from './journal.js'
import { readWorkspaceLedger, writeWorkspaceLedger } from './ledger.js'
import type { BioPleaseProject, BioPleaseSnapshot, RecentProjectRecord } from './types.js'
import { ensureProjectWorkspace, loadProject, normalizeProjectSlug, resolveRepoRoot } from './workspace.js'

type RecentProjectsFile = {
  projects: RecentProjectRecord[]
}

function getAppStateDirectory(): string {
  return join(resolveRepoRoot(), '.bioplease-app')
}

function getRecentProjectsPath(): string {
  return join(getAppStateDirectory(), 'recent-projects.json')
}

async function readRecentProjectsFile(): Promise<RecentProjectsFile> {
  try {
    return JSON.parse(await readFile(getRecentProjectsPath(), 'utf8')) as RecentProjectsFile
  } catch {
    return { projects: [] }
  }
}

async function writeRecentProjectsFile(file: RecentProjectsFile): Promise<void> {
  await mkdir(getAppStateDirectory(), { recursive: true })
  await writeFile(getRecentProjectsPath(), `${JSON.stringify(file, null, 2)}\n`, 'utf8')
}

export async function rememberProject(project: BioPleaseProject): Promise<void> {
  const file = await readRecentProjectsFile()
  const nextRecord: RecentProjectRecord = {
    root: project.root,
    slug: normalizeProjectSlug(project.config.slug),
    title: project.config.title,
    updatedAt: new Date().toISOString(),
    question: project.config.researchQuestion,
  }

  const deduped = file.projects.filter(record => resolve(record.root) !== project.root)
  deduped.unshift(nextRecord)
  file.projects = deduped.slice(0, 24)
  await writeRecentProjectsFile(file)
}

export async function listRecentProjects(): Promise<RecentProjectRecord[]> {
  const file = await readRecentProjectsFile()
  return file.projects.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
}

export async function openProjectFolder(params: {
  projectRoot: string
  title?: string
  researchQuestion?: string
  backgroundContext?: string
}): Promise<BioPleaseProject> {
  const project = await ensureProjectWorkspace({
    projectRoot: params.projectRoot,
    title: params.title,
    researchQuestion: params.researchQuestion,
    backgroundContext: params.backgroundContext,
  })
  await rememberProject(project)
  return project
}

export async function loadProjectByRoot(projectRoot: string): Promise<BioPleaseProject> {
  return loadProject(projectRoot)
}

export async function getProjectSnapshot(
  projectRoot: string,
): Promise<BioPleaseSnapshot> {
  const project = await loadProject(projectRoot)
  await rememberProject(project)

  const [plan, state, journal, summary, artifacts, enabledTools] = await Promise.all([
    readFile(project.paths.plan, 'utf8').catch(() => ''),
    readRuntimeState(project),
    readJournal(project, 200),
    readSessionSummary(project),
    buildArtifactManifest(project),
    readEnabledTools(project),
  ])

  await writeWorkspaceLedger(project)
  const ledger = await readWorkspaceLedger(project)

  return {
    project: {
      root: project.root,
      config: project.config,
    },
    plan,
    state: {
      ...state,
      recentFiles: ledger.recentFiles,
      recentArtifacts: artifacts.slice(0, 12).map(record => record.relativePath),
      enabledTools,
    },
    journal,
    summary,
    artifacts,
    enabledTools,
  }
}
