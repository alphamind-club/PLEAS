export const BIOPLEASE_FRAMEWORK_VERSION = '2.0.0'

export const BIOPLEASE_PHASES = [
  'Plan',
  'Learn',
  'Execute',
  'Assess',
  'Share',
] as const

export type BioPleasePhase = (typeof BIOPLEASE_PHASES)[number]

export type BioPleasePhaseStatus =
  | 'idle'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'skipped'

export type BioPleaseTaskStatus =
  | 'todo'
  | 'in_progress'
  | 'done'
  | 'blocked'

export type BioPleaseProviderProfile = {
  model: string
  baseUrl: string
  apiKey: string
}

export type BioPleasePhaseModelRouting = {
  Plan: string
  Learn: string
  Execute: string
  Assess: string
  Share: string
  fallback: string
}

export type BioPleaseConfig = {
  slug: string
  title: string
  researchQuestion: string
  backgroundContext: string
  frameworkVersion: string
  createdAt: string
  updatedAt: string
  localWorkspaceMode: 'folder-root'
  metadataDirectory: '.bioplease'
  phaseModelRouting: BioPleasePhaseModelRouting
  bioContext: {
    autoInstall: boolean
    registryUrl: string
    cacheTtlMs: number
    defaultKnowledgebaseUrl: string
  }
  toolCatalogRoots: string[]
}

export type BioPleasePaths = {
  metadataRoot: string
  cache: string
  tooling: string
  summaries: string
  transcripts: string
  memory: string
  plans: string
  config: string
  state: string
  plan: string
  planHistory: string
  journal: string
  artifactManifest: string
  currentTask: string
  reports: string
  artifacts: string
  figures: string
  data: string
  settingsLocal: string
  skillsRoot: string
  projectClaude: string
  mcpConfig: string
  sessionSummary: string
}

export type BioPleaseProject = {
  root: string
  repoRoot: string
  config: BioPleaseConfig
  paths: BioPleasePaths
}

export type ArtifactKind =
  | 'script'
  | 'notebook'
  | 'data'
  | 'figure'
  | 'report'
  | 'paper'
  | 'log'
  | 'json'
  | 'other'

export type ArtifactRecord = {
  relativePath: string
  kind: ArtifactKind
  sizeBytes: number
  modifiedAt: string
}

export type WorkspaceLedger = {
  generatedAt: string
  projectRoot: string
  currentWorkingDirectory: string
  activeTaskSummary: string | null
  recentFiles: string[]
  primaryScripts: string[]
  datasets: string[]
  figures: string[]
  outputs: string[]
  failedCommands: string[]
  unfinishedTasks: string[]
  artifactManifestPath: string
  reportStatus: {
    planExists: boolean
    summaryExists: boolean
    sessionSummaryExists: boolean
  }
  fileCounts: {
    total: number
    hidden: number
    visible: number
    data: number
    figures: number
    reports: number
    artifacts: number
  }
}

export type BioPleaseProgressItem = {
  id: string
  title: string
  status: BioPleaseTaskStatus
  phase: BioPleasePhase | 'General'
  detail?: string
  lineNumber?: number
}

export type BioPleasePhaseState = {
  phase: BioPleasePhase
  status: BioPleasePhaseStatus
  model: string
  startedAt?: string
  completedAt?: string
  summary?: string
  error?: string
  sessionId?: string
}

export type BioPleaseToolRecord = {
  id: string
  name: string
  description: string
  source: 'biocontext' | 'local-bundle'
  enabledAt: string
  remote: boolean
  url?: string
  tags: string[]
  provenance?: string
}

export type BioPleaseSessionRecord = {
  id: string
  goal: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  startedAt: string
  completedAt?: string
  error?: string
  phases: BioPleasePhaseState[]
  latestSummary?: string
  transcripts: string[]
}

export type BioPleaseRuntimeState = {
  version: string
  projectRoot: string
  updatedAt: string
  activeGoal: string | null
  activePhase: BioPleasePhase | null
  activeSessionId: string | null
  latestSummary: string
  progressItems: BioPleaseProgressItem[]
  phaseStates: BioPleasePhaseState[]
  recentFiles: string[]
  recentArtifacts: string[]
  previewUrls: string[]
  lastToolEvent: string | null
  blockedReason: string | null
  enabledTools: BioPleaseToolRecord[]
  sessions: BioPleaseSessionRecord[]
}

export type RunJournalEvent = {
  timestamp: string
  type: string
  sessionId?: string
  phase?: BioPleasePhase
  detail: Record<string, unknown>
}

export type BioPleaseEvent = {
  id: string
  type:
    | 'phase_started'
    | 'phase_completed'
    | 'phase_failed'
    | 'phase_output'
    | 'plan_updated'
    | 'task_progress'
    | 'tool_enabled'
    | 'tool_revoked'
    | 'tool_started'
    | 'tool_progress'
    | 'tool_finished'
    | 'artifact_created'
    | 'summary_updated'
    | 'run_started'
    | 'run_completed'
    | 'run_failed'
    | 'run_cancelled'
  timestamp: string
  projectRoot: string
  sessionId?: string
  phase?: BioPleasePhase
  message?: string
  data?: Record<string, unknown>
}

export type RuntimeCheck = {
  name: string
  available: boolean
  command: string
  path: string | null
  detail: string
}

export type DoctorReport = {
  generatedAt: string
  checks: RuntimeCheck[]
}

export type BioPleaseSnapshot = {
  project: {
    root: string
    config: BioPleaseConfig
  }
  plan: string
  state: BioPleaseRuntimeState
  journal: RunJournalEvent[]
  summary: string
  artifacts: ArtifactRecord[]
  enabledTools: BioPleaseToolRecord[]
}

export type RecentProjectRecord = {
  root: string
  slug: string
  title: string
  updatedAt: string
  question: string
}
