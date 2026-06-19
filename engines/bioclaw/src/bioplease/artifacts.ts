import { readdir, stat, writeFile } from 'fs/promises'
import { join, relative } from 'path'

import type { ArtifactKind, ArtifactRecord, BioPleaseProject } from './types.js'

const EXTENSION_KIND_MAP: Record<string, ArtifactKind> = {
  '.py': 'script',
  '.r': 'script',
  '.sh': 'script',
  '.ps1': 'script',
  '.ts': 'script',
  '.tsx': 'script',
  '.js': 'script',
  '.jsx': 'script',
  '.ipynb': 'notebook',
  '.csv': 'data',
  '.tsv': 'data',
  '.parquet': 'data',
  '.h5': 'data',
  '.h5ad': 'data',
  '.npz': 'data',
  '.pkl': 'data',
  '.json': 'json',
  '.jsonl': 'json',
  '.png': 'figure',
  '.jpg': 'figure',
  '.jpeg': 'figure',
  '.svg': 'figure',
  '.pdf': 'paper',
  '.md': 'report',
  '.txt': 'log',
  '.log': 'log',
  '.html': 'other',
}

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

function classifyArtifactKind(path: string): ArtifactKind {
  const normalized = path.toLowerCase()
  const extension = normalized.includes('.')
    ? normalized.slice(normalized.lastIndexOf('.'))
    : ''
  if (normalized.endsWith('/paper.md') || normalized.endsWith('\\paper.md')) {
    return 'paper'
  }
  return EXTENSION_KIND_MAP[extension] ?? 'other'
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

export async function buildArtifactManifest(
  project: BioPleaseProject,
): Promise<ArtifactRecord[]> {
  const files = await scanFiles(project.root)
  const records = await Promise.all(
    files.map(async file => {
      const info = await stat(file)
      return {
        relativePath: relative(project.root, file).replaceAll('\\', '/'),
        kind: classifyArtifactKind(file),
        sizeBytes: info.size,
        modifiedAt: info.mtime.toISOString(),
      } satisfies ArtifactRecord
    }),
  )

  records.sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt))
  await writeFile(
    project.paths.artifactManifest,
    `${JSON.stringify(records, null, 2)}\n`,
    'utf8',
  )
  return records
}
