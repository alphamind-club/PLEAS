import { appendFile, readFile, writeFile } from 'fs/promises'
import { join } from 'path'

import type {
  BioPleaseRuntimeState,
  BioPleaseToolRecord,
  BioPleaseProject,
  RunJournalEvent,
} from './types.js'

function createEmptyRuntimeState(project: BioPleaseProject): BioPleaseRuntimeState {
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

export async function readRuntimeState(
  project: BioPleaseProject,
): Promise<BioPleaseRuntimeState> {
  try {
    return JSON.parse(await readFile(project.paths.state, 'utf8')) as BioPleaseRuntimeState
  } catch {
    const fallback = createEmptyRuntimeState(project)
    await writeRuntimeState(project, fallback)
    return fallback
  }
}

export async function writeRuntimeState(
  project: BioPleaseProject,
  state: BioPleaseRuntimeState,
): Promise<void> {
  const next = {
    ...state,
    version: project.config.frameworkVersion,
    projectRoot: project.root,
    updatedAt: new Date().toISOString(),
  }
  await writeFile(project.paths.state, `${JSON.stringify(next, null, 2)}\n`, 'utf8')
}

export async function updateRuntimeState(
  project: BioPleaseProject,
  updater: (state: BioPleaseRuntimeState) => BioPleaseRuntimeState,
): Promise<BioPleaseRuntimeState> {
  const current = await readRuntimeState(project)
  const next = updater(current)
  await writeRuntimeState(project, next)
  return next
}

export async function readEnabledTools(
  project: BioPleaseProject,
): Promise<BioPleaseToolRecord[]> {
  const state = await readRuntimeState(project)
  if (state.enabledTools.length > 0) {
    return state.enabledTools
  }

  try {
    return JSON.parse(
      await readFile(join(project.paths.tooling, 'enabled-tools.json'), 'utf8'),
    ) as BioPleaseToolRecord[]
  } catch {
    return []
  }
}

export async function writeEnabledTools(
  project: BioPleaseProject,
  tools: BioPleaseToolRecord[],
): Promise<void> {
  await writeFile(
    join(project.paths.tooling, 'enabled-tools.json'),
    `${JSON.stringify(tools, null, 2)}\n`,
    'utf8',
  )
  await updateRuntimeState(project, state => ({
    ...state,
    enabledTools: tools,
  }))
}

export async function appendJournalEvent(
  project: BioPleaseProject,
  type: string,
  detail: Record<string, unknown>,
  options?: {
    sessionId?: string
    phase?: RunJournalEvent['phase']
  },
): Promise<RunJournalEvent> {
  const event: RunJournalEvent = {
    timestamp: new Date().toISOString(),
    type,
    detail,
    ...(options?.sessionId ? { sessionId: options.sessionId } : {}),
    ...(options?.phase ? { phase: options.phase } : {}),
  }
  await appendFile(project.paths.journal, `${JSON.stringify(event)}\n`, 'utf8')
  return event
}

export async function readJournal(
  project: BioPleaseProject,
  limit = 100,
): Promise<RunJournalEvent[]> {
  try {
    const raw = await readFile(project.paths.journal, 'utf8')
    return raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(Boolean)
      .slice(-limit)
      .map(line => JSON.parse(line) as RunJournalEvent)
  } catch {
    return []
  }
}

export async function appendTranscriptEvent(
  project: BioPleaseProject,
  sessionId: string,
  payload: Record<string, unknown>,
): Promise<string> {
  const path = join(project.paths.transcripts, `${sessionId}.jsonl`)
  await appendFile(path, `${JSON.stringify(payload)}\n`, 'utf8')
  return path
}

export async function readTranscript(
  project: BioPleaseProject,
  sessionId: string,
): Promise<Array<Record<string, unknown>>> {
  try {
    const raw = await readFile(join(project.paths.transcripts, `${sessionId}.jsonl`), 'utf8')
    return raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(Boolean)
      .map(line => JSON.parse(line) as Record<string, unknown>)
  } catch {
    return []
  }
}

export async function overwriteSessionSummary(
  project: BioPleaseProject,
  markdown: string,
): Promise<void> {
  await writeFile(project.paths.sessionSummary, markdown, 'utf8')
}

export async function readSessionSummary(project: BioPleaseProject): Promise<string> {
  try {
    return await readFile(project.paths.sessionSummary, 'utf8')
  } catch {
    return ''
  }
}

export async function appendPlanHistory(
  project: BioPleaseProject,
  plan: string,
  options?: {
    actor?: string
    sessionId?: string
  },
): Promise<void> {
  await appendFile(
    project.paths.planHistory,
    `${JSON.stringify({
      timestamp: new Date().toISOString(),
      actor: options?.actor ?? 'system',
      sessionId: options?.sessionId ?? null,
      plan,
    })}\n`,
    'utf8',
  )
}
