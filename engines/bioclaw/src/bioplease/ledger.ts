import { readdir, readFile, stat, writeFile } from 'fs/promises'
import { join, relative } from 'path'

import { buildArtifactManifest } from './artifacts.js'
import { readJournal } from './journal.js'
import type { ArtifactRecord, BioPleaseProject, WorkspaceLedger } from './types.js'

const EXCLUDED_DIRECTORIES = new Set([
  '.git',
  '.bioplease',
  '.bioplease-app',
  '.claude',
  'node_modules',
  'dist',
  'coverage',
  '.next',
  '.turbo',
  '.idea',
  '.vscode',
  'projects',
  'BioPlease_tools',
])

function getLedgerPath(project: BioPleaseProject): string {
  return join(project.paths.cache, 'workspace-ledger.json')
}

async function readOptionalFile(path: string): Promise<string> {
  try {
    return await readFile(path, 'utf8')
  } catch {
    return ''
  }
}

async function scanFiles(root: string): Promise<string[]> {
  const files: string[] = []

  async function walk(current: string): Promise<void> {
    const entries = await readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      if (EXCLUDED_DIRECTORIES.has(entry.name)) {
        continue
      }
      const fullPath = join(current, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath)
        continue
      }
      if (entry.isFile()) {
        files.push(fullPath)
      }
    }
  }

  await walk(root)
  return files
}

function toRelative(project: BioPleaseProject, path: string): string {
  return relative(project.root, path).replaceAll('\\', '/')
}

function summarizeActiveTask(content: string): string | null {
  const goalLine = content
    .split(/\r?\n/)
    .map(line => line.trim())
    .find(line => line.toLowerCase().startsWith('goal:'))
  return goalLine ? goalLine.slice('goal:'.length).trim() : null
}

function extractUnfinishedTasks(plan: string): string[] {
  return plan
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => /^\-\s\[\s\]/.test(line))
    .map(line => line.replace(/^\-\s\[\s\]\s*/, ''))
    .slice(0, 12)
}

function pickTopArtifacts(records: ArtifactRecord[], kind: ArtifactRecord['kind'], limit = 10): string[] {
  return records
    .filter(record => record.kind === kind)
    .slice(0, limit)
    .map(record => record.relativePath)
}

export async function collectWorkspaceLedger(
  project: BioPleaseProject,
): Promise<WorkspaceLedger> {
  const [artifactManifest, files, plan, currentTask, journal] = await Promise.all([
    buildArtifactManifest(project),
    scanFiles(project.root),
    readOptionalFile(project.paths.plan),
    readOptionalFile(project.paths.currentTask),
    readJournal(project, 200),
  ])

  const recentFiles = await Promise.all(
    files.map(async path => ({
      path,
      mtimeMs: (await stat(path)).mtimeMs,
    })),
  )
  recentFiles.sort((left, right) => right.mtimeMs - left.mtimeMs)

  const reportFiles = artifactManifest.filter(record => record.relativePath.startsWith('reports/'))
  const outputFiles = artifactManifest.filter(
    record =>
      record.relativePath.startsWith('reports/') ||
      record.relativePath.startsWith('artifacts/') ||
      record.relativePath.startsWith('figures/') ||
      record.relativePath.startsWith('data/'),
  )

  return {
    generatedAt: new Date().toISOString(),
    projectRoot: project.root,
    currentWorkingDirectory: project.root,
    activeTaskSummary: summarizeActiveTask(currentTask),
    recentFiles: recentFiles.slice(0, 18).map(entry => toRelative(project, entry.path)),
    primaryScripts: [
      ...pickTopArtifacts(artifactManifest, 'script', 8),
      ...pickTopArtifacts(artifactManifest, 'notebook', 4),
    ].slice(0, 10),
    datasets: pickTopArtifacts(artifactManifest, 'data', 10),
    figures: pickTopArtifacts(artifactManifest, 'figure', 10),
    outputs: outputFiles.slice(0, 12).map(record => record.relativePath),
    failedCommands: journal
      .filter(event => /failed|warning/.test(event.type))
      .map(event => String(event.detail.message ?? event.detail.error ?? event.type))
      .slice(-6),
    unfinishedTasks: extractUnfinishedTasks(plan),
    artifactManifestPath: '.bioplease/artifact-manifest.json',
    reportStatus: {
      planExists: plan.trim().length > 0,
      summaryExists: reportFiles.length > 0,
      sessionSummaryExists: (await readOptionalFile(project.paths.sessionSummary)).trim().length > 0,
    },
    fileCounts: {
      total: files.length,
      hidden: 0,
      visible: files.length,
      data: artifactManifest.filter(record => record.kind === 'data').length,
      figures: artifactManifest.filter(record => record.kind === 'figure').length,
      reports: artifactManifest.filter(
        record => record.kind === 'report' || record.kind === 'paper',
      ).length,
      artifacts: artifactManifest.length,
    },
  }
}

export async function writeWorkspaceLedger(
  project: BioPleaseProject,
): Promise<WorkspaceLedger> {
  const ledger = await collectWorkspaceLedger(project)
  await writeFile(getLedgerPath(project), `${JSON.stringify(ledger, null, 2)}\n`, 'utf8')
  return ledger
}

export async function readWorkspaceLedger(
  project: BioPleaseProject,
): Promise<WorkspaceLedger> {
  try {
    return JSON.parse(await readFile(getLedgerPath(project), 'utf8')) as WorkspaceLedger
  } catch {
    return writeWorkspaceLedger(project)
  }
}
