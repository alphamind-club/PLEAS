import { readFile, readdir, stat, writeFile } from 'fs/promises'
import { spawn } from 'node:child_process'
import { randomUUID } from 'crypto'
import { dirname, join, relative } from 'path'
import * as readline from 'node:readline'

import { autoInstallRelevantTools } from './biocontext.js'
import { buildArtifactManifest } from './artifacts.js'
import { resolveWorkingRipgrepPath } from './doctor.js'
import {
  appendJournalEvent,
  appendPlanHistory,
  appendTranscriptEvent,
  overwriteSessionSummary,
  readEnabledTools,
  readRuntimeState,
  updateRuntimeState,
} from './journal.js'
import { readWorkspaceLedger, writeWorkspaceLedger } from './ledger.js'
import {
  buildCurrentTaskTemplate,
  buildPhaseAppendPrompt,
  buildPhaseUserPrompt,
  buildSessionSummaryMarkdown,
} from './prompts.js'
import type {
  BioPleaseEvent,
  BioPleasePhase,
  BioPleasePhaseState,
  BioPleaseProject,
  BioPleaseSessionRecord,
  BioPleaseTaskStatus,
} from './types.js'
import { BIOPLEASE_PHASES } from './types.js'
import { ensureProjectWorkspace, resolveProviderProfiles } from './workspace.js'

type Snapshot = Map<string, { size: number; mtimeMs: number }>

export type RunOptions = {
  projectRoot: string
  title?: string
  researchQuestion?: string
  backgroundContext?: string
  goal: string
  maxTurns?: number
  permissionMode?: string
  signal?: AbortSignal
  onEvent?: (event: BioPleaseEvent) => void | Promise<void>
}

export type RunSummary = {
  sessionId: string
  projectRoot: string
  transcriptPath: string
  latestSummary: string
  phaseStates: BioPleasePhaseState[]
  createdFiles: string[]
  modifiedFiles: string[]
  deletedFiles: string[]
}

type PhaseRunResult = {
  model: string
  sessionId: string | null
  resultText: string
  summaryText: string
  toolSummaries: string[]
}

type PermissionDenialRecord = {
  toolName: string
  toolUseId: string | null
  inputSummary: string
}

class BioPleasePhaseError extends Error {
  phase: BioPleasePhase
  model: string
  exitCode: number
  openClaudeSessionId: string | null
  stderrTail: string[]
  resultText: string
  assistantNote: string
  errorKind: 'auth' | 'result' | 'stderr' | 'exit_code'
  stopReason: string
  numTurns: number | null
  resultErrors: string[]
  permissionDenials: PermissionDenialRecord[]

  constructor(params: {
    phase: BioPleasePhase
    model: string
    message: string
    exitCode: number
    openClaudeSessionId: string | null
    stderrTail: string[]
    resultText: string
    assistantNote: string
    errorKind: 'auth' | 'result' | 'stderr' | 'exit_code'
    stopReason: string
    numTurns: number | null
    resultErrors: string[]
    permissionDenials: PermissionDenialRecord[]
  }) {
    super(params.message)
    this.name = 'BioPleasePhaseError'
    this.phase = params.phase
    this.model = params.model
    this.exitCode = params.exitCode
    this.openClaudeSessionId = params.openClaudeSessionId
    this.stderrTail = params.stderrTail
    this.resultText = params.resultText
    this.assistantNote = params.assistantNote
    this.errorKind = params.errorKind
    this.stopReason = params.stopReason
    this.numTurns = params.numTurns
    this.resultErrors = params.resultErrors
    this.permissionDenials = params.permissionDenials
  }
}

type OpenClaudeCliInvocation = {
  command: string
  argsPrefix: string[]
}

function extractAssistantText(payload: Record<string, unknown>): string {
  const message = payload.message as
    | { content?: Array<Record<string, unknown>> }
    | undefined
  if (!message?.content) {
    return ''
  }
  return message.content
    .filter(block => block.type === 'text' && typeof block.text === 'string')
    .map(block => String(block.text))
    .join('\n')
    .trim()
}

function getMessageContentBlocks(payload: Record<string, unknown>): Array<Record<string, unknown>> {
  const message = payload.message as
    | { content?: Array<Record<string, unknown>> }
    | undefined
  return Array.isArray(message?.content) ? message.content : []
}

function summarizeToolInput(input: unknown): string {
  if (!input || typeof input !== 'object') {
    return ''
  }

  const record = input as Record<string, unknown>
  const filePath =
    typeof record.file_path === 'string'
      ? record.file_path
      : typeof record.path === 'string'
        ? record.path
        : ''
  if (filePath) {
    return sanitizeStreamText(filePath)
  }

  if (typeof record.command === 'string' && record.command) {
    return sanitizeStreamText(record.command).replace(/\s+/g, ' ').slice(0, 180)
  }

  if (typeof record.query === 'string' && record.query) {
    return sanitizeStreamText(record.query)
  }

  if (typeof record.url === 'string' && record.url) {
    return sanitizeStreamText(record.url)
  }

  return sanitizeStreamText(JSON.stringify(record)).slice(0, 180)
}

function extractAssistantToolUses(payload: Record<string, unknown>): Array<{
  toolUseId: string | null
  toolName: string
  inputSummary: string
}> {
  return getMessageContentBlocks(payload)
    .filter(block => block.type === 'tool_use')
    .map(block => ({
      toolUseId: typeof block.id === 'string' ? block.id : null,
      toolName: sanitizeStreamText(String(block.name ?? 'Tool')),
      inputSummary: summarizeToolInput(block.input),
    }))
}

function summarizeToolResultContent(content: string): string {
  const trimmed = sanitizeStreamText(content).trim()
  if (!trimmed) {
    return ''
  }

  const normalized = trimmed.replace(/\s+/g, ' ')
  return normalized.length > 220 ? `${normalized.slice(0, 217)}...` : normalized
}

function summarizePermissionDenial(record: Record<string, unknown>): PermissionDenialRecord {
  return {
    toolName: sanitizeStreamText(String(record.tool_name ?? 'Tool')),
    toolUseId: typeof record.tool_use_id === 'string' ? record.tool_use_id : null,
    inputSummary: summarizeToolInput(record.tool_input),
  }
}

function buildResultMessage(params: {
  resultText: string
  resultErrors: string[]
  permissionDenials: PermissionDenialRecord[]
  stopReason: string
  numTurns: number | null
}): string {
  if (params.resultText) {
    return params.resultText
  }

  const parts: string[] = []

  if (params.resultErrors.length > 0) {
    parts.push(params.resultErrors.join('; '))
  }

  if (params.permissionDenials.length > 0) {
    const deniedTools = Array.from(
      new Set(
        params.permissionDenials
          .map(record => record.toolName)
          .filter(Boolean),
      ),
    )
    const toolLabel =
      deniedTools.length > 0 ? deniedTools.join(', ') : 'one or more tools'
    parts.push(
      `${params.permissionDenials.length} tool call${
        params.permissionDenials.length === 1 ? '' : 's'
      } were denied (${toolLabel})`,
    )
  }

  if (params.stopReason) {
    parts.push(`stop reason: ${params.stopReason}`)
  }

  if (typeof params.numTurns === 'number' && Number.isFinite(params.numTurns)) {
    parts.push(`turns: ${params.numTurns}`)
  }

  return parts.join(' | ').trim()
}

function extractUserToolResults(payload: Record<string, unknown>): Array<{
  toolUseId: string
  summary: string
  isError: boolean
}> {
  return getMessageContentBlocks(payload)
    .filter(block => block.type === 'tool_result' && typeof block.tool_use_id === 'string')
    .map(block => {
      const content =
        typeof block.content === 'string'
          ? block.content
          : Array.isArray(block.content)
            ? block.content
                .map(item =>
                  typeof item === 'string'
                    ? item
                    : typeof item === 'object' && item && 'text' in item
                      ? String((item as { text?: unknown }).text ?? '')
                      : '',
                )
                .filter(Boolean)
                .join('\n')
            : ''

      return {
        toolUseId: String(block.tool_use_id),
        summary: summarizeToolResultContent(content),
        isError: block.is_error === true,
      }
    })
}

function sanitizeStreamText(value: string): string {
  return value
    .replace(/sk-[A-Za-z0-9*_-]{8,}/g, 'sk-...redacted...')
    .replace(/\u00a0/g, ' ')
}

function getStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter(item => typeof item === 'string').map(item => String(item))
    : []
}

function terminateChildProcess(child: ReturnType<typeof spawn>): void {
  if (child.exitCode !== null || child.killed) {
    return
  }

  try {
    child.kill()
  } catch {}

  if (process.platform === 'win32' && child.pid) {
    const killer = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], {
      stdio: 'ignore',
      windowsHide: true,
    })
    killer.on('error', () => {})
    killer.unref()
    return
  }

  const timer = setTimeout(() => {
    try {
      child.kill('SIGKILL')
    } catch {}
  }, 250)
  timer.unref()
}

async function readOptionalFile(path: string): Promise<string> {
  try {
    return await readFile(path, 'utf8')
  } catch {
    return ''
  }
}

async function scanSnapshot(root: string): Promise<Snapshot> {
  const snapshot: Snapshot = new Map()

  async function walk(current: string): Promise<void> {
    const entries = await readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      if (['.git', 'node_modules', '.next', 'dist', 'coverage'].includes(entry.name)) {
        continue
      }
      const fullPath = join(current, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath)
        continue
      }
      if (!entry.isFile()) {
        continue
      }
      const info = await stat(fullPath)
      snapshot.set(fullPath, { size: info.size, mtimeMs: info.mtimeMs })
    }
  }

  await walk(root)
  return snapshot
}

function diffSnapshots(
  project: BioPleaseProject,
  before: Snapshot,
  after: Snapshot,
): { created: string[]; modified: string[]; deleted: string[] } {
  const created: string[] = []
  const modified: string[] = []
  const deleted: string[] = []

  for (const [file, current] of after.entries()) {
    const previous = before.get(file)
    const relativePath = relative(project.root, file).replaceAll('\\', '/')
    if (!previous) {
      created.push(relativePath)
      continue
    }
    if (previous.size !== current.size || previous.mtimeMs !== current.mtimeMs) {
      modified.push(relativePath)
    }
  }

  for (const file of before.keys()) {
    if (!after.has(file)) {
      deleted.push(relative(project.root, file).replaceAll('\\', '/'))
    }
  }

  created.sort()
  modified.sort()
  deleted.sort()

  return { created, modified, deleted }
}

function createEvent(
  project: BioPleaseProject,
  type: BioPleaseEvent['type'],
  message: string,
  data?: Record<string, unknown>,
  options?: { sessionId?: string; phase?: BioPleasePhase },
): BioPleaseEvent {
  return {
    id: randomUUID(),
    type,
    timestamp: new Date().toISOString(),
    projectRoot: project.root,
    message,
    data,
    ...(options?.sessionId ? { sessionId: options.sessionId } : {}),
    ...(options?.phase ? { phase: options.phase } : {}),
  }
}

async function emitEvent(
  project: BioPleaseProject,
  callback: RunOptions['onEvent'],
  type: BioPleaseEvent['type'],
  message: string,
  data?: Record<string, unknown>,
  options?: { sessionId?: string; phase?: BioPleasePhase },
): Promise<void> {
  await callback?.(createEvent(project, type, message, data, options))
}

function parsePlanProgress(plan: string) {
  const lines = plan.split(/\r?\n/)
  return lines
    .map((line, index) => ({ line, index }))
    .filter(({ line }) => /^\-\s\[( |x)\]/i.test(line.trim()))
    .map(({ line, index }) => {
      const trimmed = line.trim()
      const checked = /^\-\s\[x\]/i.test(trimmed)
      const title = trimmed.replace(/^\-\s\[[ x]\]\s*/i, '')
      return {
        id: `plan-${index + 1}`,
        title,
        status: (checked ? 'done' : 'todo') as BioPleaseTaskStatus,
        phase: 'General' as const,
        lineNumber: index + 1,
      }
    })
}

function buildProviderEnv(model: string): NodeJS.ProcessEnv {
  const profiles = resolveProviderProfiles()
  const profile = profiles[model]
  if (!profile) {
    throw new Error(`No provider profile configured for model ${model}`)
  }

  return {
    ...process.env,
    CLAUDE_CODE_USE_OPENAI: '1',
    OPENAI_BASE_URL: profile.baseUrl,
    OPENAI_API_KEY: profile.apiKey,
    OPENAI_MODEL: profile.model,
  }
}

function resolveBunExecutable(): string {
  return process.execPath.toLowerCase().includes('bun')
    ? process.execPath
    : Bun.which('bun') ?? 'C:\\Users\\tpan6\\.bun\\bin\\bun.exe'
}

function resolveNodeExecutable(): string {
  return Bun.which('node') ?? 'node'
}

async function ensurePackagedOpenClaudeCli(params: {
  project: BioPleaseProject
  orchestrationSessionId: string
  onEvent?: RunOptions['onEvent']
}): Promise<OpenClaudeCliInvocation> {
  const distCliPath = join(params.project.repoRoot, 'dist', 'cli.mjs')
  const shouldForceBuild = process.env.BIOPLEASE_FORCE_OPENCLAUDE_BUILD === '1'

  if (!shouldForceBuild) {
    try {
      await stat(distCliPath)
      return {
        command: resolveNodeExecutable(),
        argsPrefix: [distCliPath],
      }
    } catch {}
  }

  await emitEvent(
    params.project,
    params.onEvent,
    'task_progress',
    'Building packaged OpenClaude runtime',
    { step: 'openclaude_build' },
    { sessionId: params.orchestrationSessionId, phase: 'Plan' },
  )

  const buildOutput: string[] = []
  const build = spawn(resolveBunExecutable(), ['run', 'build'], {
    cwd: params.project.repoRoot,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  const stdoutRl = readline.createInterface({
    input: build.stdout,
    crlfDelay: Infinity,
  })
  const stderrRl = readline.createInterface({
    input: build.stderr,
    crlfDelay: Infinity,
  })

  stdoutRl.on('line', line => {
    buildOutput.push(line)
  })
  stderrRl.on('line', line => {
    buildOutput.push(line)
  })

  const exitCode = await new Promise<number>((resolve, reject) => {
    build.once('error', reject)
    build.once('close', resolve)
  })

  if (exitCode !== 0) {
    throw new Error(
      buildOutput.at(-1) || `OpenClaude build failed with exit code ${exitCode}`,
    )
  }

  try {
    await stat(distCliPath)
  } catch {
    throw new Error('OpenClaude build completed without producing dist/cli.mjs')
  }

  await emitEvent(
    params.project,
    params.onEvent,
    'task_progress',
    'Packaged OpenClaude runtime is ready',
    { step: 'openclaude_build_complete' },
    { sessionId: params.orchestrationSessionId, phase: 'Plan' },
  )

  return {
    command: resolveNodeExecutable(),
    argsPrefix: [distCliPath],
  }
}

function buildPhaseArgs(
  project: BioPleaseProject,
  phase: BioPleasePhase,
  promptPath: string,
  goal: string,
  maxTurns: number,
  permissionMode: string,
): string[] {
  return [
    '--print',
    '--output-format',
    'stream-json',
    '--verbose',
    '--bare',
    '--settings',
    project.paths.settingsLocal,
    '--mcp-config',
    project.paths.mcpConfig,
    '--add-dir',
    project.root,
    '--permission-mode',
    permissionMode,
    '--append-system-prompt-file',
    promptPath,
    '--max-turns',
    String(maxTurns),
    buildPhaseUserPrompt({
      config: project.config,
      phase,
      goal,
    }),
  ]
}

function ensureNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error('Run cancelled')
  }
}

async function runSinglePhase(params: {
  project: BioPleaseProject
  cliInvocation: OpenClaudeCliInvocation
  orchestrationSessionId: string
  phase: BioPleasePhase
  model: string
  goal: string
  permissionMode: string
  maxTurns: number
  signal?: AbortSignal
  onEvent?: RunOptions['onEvent']
}): Promise<PhaseRunResult> {
  ensureNotAborted(params.signal)
  const ledger = await writeWorkspaceLedger(params.project)
  const state = await readRuntimeState(params.project)
  const enabledTools = await readEnabledTools(params.project)
  const currentSummary = await readOptionalFile(params.project.paths.sessionSummary)
  const appendPromptPath = join(
    params.project.paths.cache,
    `phase-${params.orchestrationSessionId}-${params.phase.toLowerCase()}.md`,
  )

  await writeFile(
    appendPromptPath,
    buildPhaseAppendPrompt({
      project: params.project,
      phase: params.phase,
      goal: params.goal,
      ledger,
      progressItems: state.progressItems,
      enabledTools,
      currentSummary,
    }),
    'utf8',
  )

  const childEnv = {
    ...buildProviderEnv(params.model),
    CLAUDE_CODE_MAX_RETRIES: process.env.BIOPLEASE_CLAUDE_CODE_MAX_RETRIES ?? '1',
    PATH: (() => {
      const rgPath = resolveWorkingRipgrepPath()
      if (!rgPath) {
        return process.env.PATH
      }
      return `${dirname(rgPath)};${process.env.PATH ?? ''}`
    })(),
    CLAUDE_COWORK_MEMORY_PATH_OVERRIDE: `${params.project.paths.memory}\\`,
    CLAUDE_COWORK_MEMORY_EXTRA_GUIDELINES:
      'Persist project facts, file locations, naming conventions, dataset provenance, preview URLs, environment findings, and repeated failures.',
  }

  const child = spawn(
    params.cliInvocation.command,
    [
      ...params.cliInvocation.argsPrefix,
      ...buildPhaseArgs(
        params.project,
        params.phase,
        appendPromptPath,
        params.goal,
        params.maxTurns,
        params.permissionMode,
      ),
    ],
    {
      cwd: params.project.root,
      env: childEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )

  const emitPhaseOutput = (
    message: string,
    data?: Record<string, unknown>,
  ): Promise<void> =>
    emitEvent(
      params.project,
      params.onEvent,
      'phase_output',
      sanitizeStreamText(message),
      data,
      { sessionId: params.orchestrationSessionId, phase: params.phase },
    )

  if (params.signal) {
    const abort = () => {
      terminateChildProcess(child)
    }
    params.signal.addEventListener('abort', abort, { once: true })
  }

  let openClaudeSessionId: string | null = null
  let resultText = ''
  let summaryText = ''
  let resultIsError = false
  let resultStopReason = ''
  let resultNumTurns: number | null = null
  let resultErrors: string[] = []
  let resultPermissionDenials: PermissionDenialRecord[] = []
  let fatalChildError = ''
  const toolSummaries: string[] = []
  const stderrLines: string[] = []
  const assistantNotes: string[] = []
  const toolUseNames = new Map<string, string>()
  const toolUseInputs = new Map<string, string>()
  const startedToolUseIds = new Set<string>()
  const finishedToolUseIds = new Set<string>()
  const pendingLineTasks = new Set<Promise<void>>()

  const stdoutRl = readline.createInterface({
    input: child.stdout,
    crlfDelay: Infinity,
  })
  const stderrRl = readline.createInterface({
    input: child.stderr,
    crlfDelay: Infinity,
  })

  const trackLineTask = (task: Promise<void>) => {
    const wrappedTask = task.catch(error => {
      const message = sanitizeStreamText(
        error instanceof Error ? error.message : String(error),
      )
      stderrLines.push(message)
    })
    pendingLineTasks.add(wrappedTask)
    wrappedTask.finally(() => {
      pendingLineTasks.delete(wrappedTask)
    })
  }

  stdoutRl.on('line', line => {
    trackLineTask(
      (async () => {
      let payload: Record<string, unknown> | null = null
      try {
        payload = JSON.parse(line) as Record<string, unknown>
      } catch {
        payload = { type: 'raw', line }
      }

      if (payload && typeof payload.session_id === 'string') {
        openClaudeSessionId = payload.session_id
      }

      await appendTranscriptEvent(params.project, params.orchestrationSessionId, {
        phase: params.phase,
        payload,
      })

      if (!payload || typeof payload.type !== 'string') {
        return
      }

      if (payload.type === 'assistant') {
        const text = sanitizeStreamText(extractAssistantText(payload)).trim()
        const toolUses = extractAssistantToolUses(payload)

        for (const toolUse of toolUses) {
          if (toolUse.toolUseId) {
            toolUseNames.set(toolUse.toolUseId, toolUse.toolName)
            toolUseInputs.set(toolUse.toolUseId, toolUse.inputSummary)
          }

          if (toolUse.toolUseId && startedToolUseIds.has(toolUse.toolUseId)) {
            continue
          }

          if (toolUse.toolUseId) {
            startedToolUseIds.add(toolUse.toolUseId)
          }

          await emitEvent(
            params.project,
            params.onEvent,
            'tool_started',
            toolUse.inputSummary
              ? `${toolUse.toolName} :: ${toolUse.inputSummary}`
              : `${toolUse.toolName} started`,
            {
              toolName: toolUse.toolName,
              toolUseId: toolUse.toolUseId,
              inputSummary: toolUse.inputSummary,
            },
            { sessionId: params.orchestrationSessionId, phase: params.phase },
          )
        }

        if (text) {
          assistantNotes.push(text)
          await emitPhaseOutput(text, {
            outputKind: 'assistant',
            source: 'openclaude',
          })
        }
        return
      }

      if (payload.type === 'user') {
        const toolResults = extractUserToolResults(payload)
        for (const toolResult of toolResults) {
          const toolName = toolUseNames.get(toolResult.toolUseId) ?? 'Tool'
          const inputSummary = toolUseInputs.get(toolResult.toolUseId) ?? ''

          if (!finishedToolUseIds.has(toolResult.toolUseId)) {
            finishedToolUseIds.add(toolResult.toolUseId)
          }

          await emitEvent(
            params.project,
            params.onEvent,
            'tool_finished',
            toolResult.summary
              ? `${toolName} :: ${toolResult.summary}`
              : `${toolName} completed`,
            {
              toolName,
              toolUseId: toolResult.toolUseId,
              inputSummary,
              status: toolResult.isError ? 'error' : 'success',
              resultPreview: toolResult.summary,
            },
            { sessionId: params.orchestrationSessionId, phase: params.phase },
          )
        }
        return
      }

      if (payload.type === 'tool_progress') {
        const toolName = sanitizeStreamText(String(payload.tool_name ?? 'Tool'))
        await emitEvent(
          params.project,
          params.onEvent,
          'tool_progress',
          `${toolName} is running`,
          {
            toolName,
            toolUseId: payload.tool_use_id,
            parentToolUseId: payload.parent_tool_use_id,
            elapsedTimeSeconds: payload.elapsed_time_seconds,
            taskId: payload.task_id,
          },
          { sessionId: params.orchestrationSessionId, phase: params.phase },
        )
        return
      }

      if (payload.type === 'tool_use_summary') {
        const summary = sanitizeStreamText(String(payload.summary ?? '').trim())
        if (summary) {
          toolSummaries.push(summary)
          for (const toolUseId of getStringArray(payload.preceding_tool_use_ids)) {
            finishedToolUseIds.add(toolUseId)
          }
          await emitEvent(
            params.project,
            params.onEvent,
            'tool_finished',
            summary,
            {
              toolUseIds: payload.preceding_tool_use_ids as unknown[],
            },
            { sessionId: params.orchestrationSessionId, phase: params.phase },
          )
        }
        return
      }

      if (payload.type === 'result') {
        resultStopReason = sanitizeStreamText(String(payload.stop_reason ?? '')).trim()
        resultNumTurns =
          typeof payload.num_turns === 'number' && Number.isFinite(payload.num_turns)
            ? payload.num_turns
            : null
        resultErrors = getStringArray(payload.errors).map(item =>
          sanitizeStreamText(item).trim(),
        ).filter(Boolean)
        resultPermissionDenials = Array.isArray(payload.permission_denials)
          ? payload.permission_denials
              .filter(
                item => typeof item === 'object' && item !== null,
              )
              .map(item => summarizePermissionDenial(item as Record<string, unknown>))
          : []
        resultText = buildResultMessage({
          resultText: sanitizeStreamText(String(payload.result ?? '').trim()),
          resultErrors,
          permissionDenials: resultPermissionDenials,
          stopReason: resultStopReason,
          numTurns: resultNumTurns,
        })
        summaryText ||= resultText
        resultIsError = payload.is_error === true
        if (resultText) {
          await emitPhaseOutput(resultText, {
            outputKind: 'result',
            isError: resultIsError,
            durationMs: payload.duration_ms,
            numTurns: payload.num_turns,
            totalCostUsd: payload.total_cost_usd,
            stopReason: payload.stop_reason,
            errors: resultErrors,
            permissionDeniedCount: resultPermissionDenials.length,
            permissionDeniedTools: resultPermissionDenials.map(record => record.toolName),
          })
        }
        return
      }

      if (payload.type === 'auth_status') {
        const authOutput = getStringArray(payload.output)
          .map(item => sanitizeStreamText(item))
          .filter(Boolean)
        const authError = sanitizeStreamText(String(payload.error ?? '')).trim()
        const authMessage =
          authOutput.join('\n').trim() ||
          authError ||
          (payload.isAuthenticating === true
            ? 'Authenticating provider credentials'
            : 'Authentication status changed')

        await emitPhaseOutput(authMessage, {
          outputKind: 'auth_status',
          isAuthenticating: payload.isAuthenticating,
        })
        return
      }

      if (payload.type === 'user' || payload.type === 'user_message_replay') {
        return
      }

      if (payload.type === 'system' && payload.subtype === 'task_started') {
        const description = sanitizeStreamText(
          String(payload.description ?? 'Task started'),
        )
        const toolUseId =
          typeof payload.tool_use_id === 'string' ? payload.tool_use_id : null
        if (toolUseId && startedToolUseIds.has(toolUseId)) {
          return
        }
        if (toolUseId) {
          startedToolUseIds.add(toolUseId)
        }
        await emitEvent(
          params.project,
          params.onEvent,
          'tool_started',
          description,
          {
            taskId: payload.task_id,
            toolUseId,
            taskType: payload.task_type,
            workflowName: payload.workflow_name,
          },
          { sessionId: params.orchestrationSessionId, phase: params.phase },
        )
        return
      }

      if (payload.type === 'system' && payload.subtype === 'init') {
        const availableTools = getStringArray(payload.tools)
        await emitPhaseOutput(
          `OpenClaude initialized for ${params.phase}.`,
          {
            outputKind: 'init',
            model: payload.model,
            cwd: payload.cwd,
            permissionMode: payload.permissionMode,
            toolsCount: availableTools.length,
            toolsPreview: availableTools.slice(0, 12),
            mcpServers: Array.isArray(payload.mcp_servers)
              ? payload.mcp_servers
                  .map(server =>
                    typeof server === 'object' && server && 'name' in server
                      ? String((server as { name?: unknown }).name ?? '')
                      : '',
                  )
                  .filter(Boolean)
              : [],
            skills: getStringArray(payload.skills).slice(0, 8),
            slashCommands: getStringArray(payload.slash_commands).slice(0, 8),
          },
        )
        return
      }

      if (
        payload.type === 'system' &&
        payload.subtype === 'api_retry' &&
        Number(payload.error_status ?? 0) === 401
      ) {
        fatalChildError ||= `Authentication failed while using ${params.model}. Check the configured API key for this routed phase.`
        await emitPhaseOutput(fatalChildError, {
          outputKind: 'api_retry',
          attempt: payload.attempt,
          maxRetries: payload.max_retries,
          errorStatus: payload.error_status,
          error: payload.error,
          retryDelayMs: payload.retry_delay_ms,
          authFailure: true,
        })
        terminateChildProcess(child)
        return
      }

      if (payload.type === 'system' && payload.subtype === 'api_retry') {
        await emitPhaseOutput(
          `OpenClaude retry ${String(payload.attempt ?? '?')}/${String(payload.max_retries ?? '?')}.`,
          {
            outputKind: 'api_retry',
            attempt: payload.attempt,
            maxRetries: payload.max_retries,
            errorStatus: payload.error_status,
            error: payload.error,
            retryDelayMs: payload.retry_delay_ms,
          },
        )
        return
      }

      if (payload.type === 'system' && payload.subtype === 'task_progress') {
        const progressMessage = sanitizeStreamText(
          String(payload.description ?? payload.summary ?? 'Task progress'),
        )
        await emitEvent(
          params.project,
          params.onEvent,
          'task_progress',
          progressMessage,
          {
            taskId: payload.task_id,
            summary: payload.summary,
            lastToolName: payload.last_tool_name,
            usage: payload.usage as Record<string, unknown>,
          },
          { sessionId: params.orchestrationSessionId, phase: params.phase },
        )
        return
      }

      if (payload.type === 'system' && payload.subtype === 'task_notification') {
        const summary = sanitizeStreamText(String(payload.summary ?? '').trim())
        if (typeof payload.tool_use_id === 'string') {
          finishedToolUseIds.add(payload.tool_use_id)
        }
        await emitEvent(
          params.project,
          params.onEvent,
          'tool_finished',
          summary || `Task ${String(payload.status ?? 'completed')}`,
          {
            taskId: payload.task_id,
            toolUseId: payload.tool_use_id,
            status: payload.status,
            outputFile: payload.output_file,
            usage: payload.usage as Record<string, unknown>,
          },
          { sessionId: params.orchestrationSessionId, phase: params.phase },
        )
        return
      }

      if (payload.type === 'system' && payload.subtype === 'post_turn_summary') {
        summaryText =
          sanitizeStreamText(
            String(payload.description ?? payload.recent_action ?? payload.status_detail ?? ''),
          ).trim() || summaryText
        await emitEvent(
          params.project,
          params.onEvent,
          'summary_updated',
          summaryText || 'Updated latest summary',
          {
            statusCategory: payload.status_category,
            title: payload.title,
            needsAction: payload.needs_action,
            artifactUrls: payload.artifact_urls as unknown[],
          },
          { sessionId: params.orchestrationSessionId, phase: params.phase },
        )
        return
      }

      await emitPhaseOutput(
        `OpenClaude emitted ${String(payload.type)}${typeof payload.subtype === 'string' ? `:${payload.subtype}` : ''}.`,
        {
          outputKind: 'raw_event',
          payloadType: payload.type,
          payloadSubtype: payload.subtype,
        },
      )
    })(),
    )
  })

  stderrRl.on('line', line => {
    trackLineTask(
      (async () => {
        const sanitizedLine = sanitizeStreamText(line)
        stderrLines.push(sanitizedLine)
        await emitPhaseOutput(sanitizedLine, {
          outputKind: 'stderr',
        })
      })(),
    )
  })

  const exitCode = await new Promise<number>((resolve, reject) => {
    child.once('error', reject)
    child.once('close', resolve)
  })

  await Promise.allSettled([...pendingLineTasks])

  ensureNotAborted(params.signal)

  if (stderrLines.length > 0) {
    await appendJournalEvent(
      params.project,
      'phase_warning',
      { lines: stderrLines.slice(-20) },
      { sessionId: params.orchestrationSessionId, phase: params.phase },
    )
  }

  if (exitCode !== 0 || resultIsError) {
    const message =
      fatalChildError ||
      stderrLines.at(-1) ||
      resultText ||
      assistantNotes.at(-1) ||
      `Phase ${params.phase} failed with exit code ${exitCode}`

    throw new BioPleasePhaseError({
      phase: params.phase,
      model: params.model,
      message,
      exitCode,
      openClaudeSessionId,
      stderrTail: stderrLines.slice(-20),
      resultText,
      assistantNote: assistantNotes.at(-1) ?? '',
      stopReason: resultStopReason,
      numTurns: resultNumTurns,
      resultErrors,
      permissionDenials: resultPermissionDenials,
      errorKind: fatalChildError
        ? 'auth'
        : resultIsError
          ? 'result'
          : stderrLines.length > 0
            ? 'stderr'
            : 'exit_code',
    })
  }

  return {
    model: params.model,
    sessionId: openClaudeSessionId,
    resultText,
    summaryText: summaryText || assistantNotes.at(-1) || resultText,
    toolSummaries,
  }
}

async function markPhaseState(
  project: BioPleaseProject,
  orchestrationSessionId: string,
  phaseState: BioPleasePhaseState,
): Promise<void> {
  await updateRuntimeState(project, state => {
    const phaseStates = state.phaseStates.map(existing =>
      existing.phase === phaseState.phase ? { ...existing, ...phaseState } : existing,
    )

    const sessions = state.sessions.map(session =>
      session.id === orchestrationSessionId
        ? {
            ...session,
            phases: session.phases.map(existing =>
              existing.phase === phaseState.phase ? { ...existing, ...phaseState } : existing,
            ),
          }
        : session,
    )

    return {
      ...state,
      activePhase:
        phaseState.status === 'running' ? phaseState.phase : state.activePhase,
      phaseStates,
      sessions,
    }
  })
}

function defaultSessionPhases(project: BioPleaseProject): BioPleasePhaseState[] {
  return [
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
  ]
}

export async function runBioPleaseSession(options: RunOptions): Promise<RunSummary> {
  const project = await ensureProjectWorkspace({
    projectRoot: options.projectRoot,
    title: options.title,
    researchQuestion: options.researchQuestion,
    backgroundContext: options.backgroundContext,
  })

  const orchestrationSessionId = randomUUID()
  const transcriptPath = join(project.paths.transcripts, `${orchestrationSessionId}.jsonl`)
  const permissionMode = options.permissionMode ?? 'bypassPermissions'
  const maxTurnsPerPhase = Math.max(8, Math.ceil((options.maxTurns ?? 45) / 5))

  const beforeSnapshot = await scanSnapshot(project.root)
  const initialPlan = await readOptionalFile(project.paths.plan)
  let lastPlan = initialPlan

  await writeFile(
    project.paths.currentTask,
    buildCurrentTaskTemplate(options.goal),
    'utf8',
  )

  const sessionRecord: BioPleaseSessionRecord = {
    id: orchestrationSessionId,
    goal: options.goal,
    status: 'running',
    startedAt: new Date().toISOString(),
    phases: defaultSessionPhases(project),
    latestSummary: 'Run started.',
    transcripts: [transcriptPath],
  }

  await updateRuntimeState(project, state => ({
    ...state,
    activeGoal: options.goal,
    activePhase: 'Plan',
    activeSessionId: orchestrationSessionId,
    blockedReason: null,
    sessions: [
      sessionRecord,
      ...state.sessions.filter(session => session.id !== orchestrationSessionId),
    ].slice(0, 20),
  }))

  await appendJournalEvent(
    project,
    'run_started',
    { goal: options.goal },
    { sessionId: orchestrationSessionId },
  )
  await emitEvent(
    project,
    options.onEvent,
    'run_started',
    options.goal,
    { goal: options.goal },
    { sessionId: orchestrationSessionId },
  )

  let latestSummary = ''
  let activePhase: BioPleasePhase | null = null
  let activeModel: string | null = null

  try {
    const cliInvocation = await ensurePackagedOpenClaudeCli({
      project,
      orchestrationSessionId,
      onEvent: options.onEvent,
    })

    for (const phase of BIOPLEASE_PHASES) {
      activePhase = phase
      ensureNotAborted(options.signal)
      let model =
        phase === 'Plan'
          ? project.config.phaseModelRouting.Plan
          : phase === 'Learn'
            ? project.config.phaseModelRouting.Learn
            : phase === 'Execute'
              ? project.config.phaseModelRouting.Execute
              : phase === 'Assess'
                ? project.config.phaseModelRouting.Assess
                : project.config.phaseModelRouting.Share
      activeModel = model

      if (phase === 'Learn') {
        const beforeTools = await readEnabledTools(project)
        const afterTools = await autoInstallRelevantTools(project, options.goal)
        const beforeIds = new Set(beforeTools.map(tool => tool.id))
        for (const tool of afterTools.filter(tool => !beforeIds.has(tool.id))) {
          await emitEvent(
            project,
            options.onEvent,
            'tool_enabled',
            tool.name,
            { toolId: tool.id, url: tool.url ?? null },
            { sessionId: orchestrationSessionId, phase },
          )
        }
      }

      const startedAt = new Date().toISOString()
      const startingState: BioPleasePhaseState = {
        phase,
        status: 'running',
        model,
        startedAt,
        completedAt: undefined,
        summary: undefined,
        error: undefined,
        sessionId: undefined,
      }

      await markPhaseState(project, orchestrationSessionId, startingState)
      await appendJournalEvent(
        project,
        'phase_started',
        { model },
        { sessionId: orchestrationSessionId, phase },
      )
      await emitEvent(
        project,
        options.onEvent,
        'phase_started',
        `${phase} started`,
        { model },
        { sessionId: orchestrationSessionId, phase },
      )

      let phaseResult: PhaseRunResult
      try {
        phaseResult = await runSinglePhase({
          project,
          cliInvocation,
          orchestrationSessionId,
          phase,
          model,
          goal: options.goal,
          permissionMode,
          maxTurns: maxTurnsPerPhase,
          signal: options.signal,
          onEvent: options.onEvent,
        })
      } catch (error) {
        if (phase === 'Execute' && model !== project.config.phaseModelRouting.fallback) {
          model = project.config.phaseModelRouting.fallback
          activeModel = model
          await appendJournalEvent(
            project,
            'phase_retry',
            {
              phase,
              fromModel: project.config.phaseModelRouting.Execute,
              toModel: model,
              reason: error instanceof Error ? error.message : String(error),
            },
            { sessionId: orchestrationSessionId, phase },
          )
          phaseResult = await runSinglePhase({
            project,
            cliInvocation,
            orchestrationSessionId,
            phase,
            model,
            goal: options.goal,
            permissionMode,
            maxTurns: maxTurnsPerPhase,
            signal: options.signal,
            onEvent: options.onEvent,
          })
        } else {
          throw error
        }
      }

      latestSummary = phaseResult.summaryText || latestSummary

      const completedAt = new Date().toISOString()
      const completedState: BioPleasePhaseState = {
        phase,
        status: 'completed',
        model,
        startedAt,
        completedAt,
        summary: phaseResult.summaryText,
        error: undefined,
        sessionId: phaseResult.sessionId ?? undefined,
      }

      await markPhaseState(project, orchestrationSessionId, completedState)

      const [planText, ledger, artifacts] = await Promise.all([
        readOptionalFile(project.paths.plan),
        readWorkspaceLedger(project),
        buildArtifactManifest(project),
      ])

      const progressItems = parsePlanProgress(planText)
      await updateRuntimeState(project, state => ({
        ...state,
        activePhase: phase === 'Share' ? null : state.activePhase,
        latestSummary: latestSummary || state.latestSummary,
        lastToolEvent: phaseResult.toolSummaries.at(-1) ?? state.lastToolEvent,
        recentFiles: ledger.recentFiles,
        recentArtifacts: artifacts.slice(0, 12).map(record => record.relativePath),
        progressItems,
      }))

      await appendJournalEvent(
        project,
        'phase_completed',
        {
          model,
          summary: phaseResult.summaryText,
          openClaudeSessionId: phaseResult.sessionId,
        },
        { sessionId: orchestrationSessionId, phase },
      )
      await emitEvent(
        project,
        options.onEvent,
        'phase_completed',
        phaseResult.summaryText || `${phase} completed`,
        { model, openClaudeSessionId: phaseResult.sessionId ?? null },
        { sessionId: orchestrationSessionId, phase },
      )

      if (planText.trim() !== lastPlan.trim()) {
        await appendPlanHistory(project, planText, {
          actor: phase,
          sessionId: orchestrationSessionId,
        })
        await emitEvent(
          project,
          options.onEvent,
          'plan_updated',
          `${phase} updated the plan`,
          { lineCount: planText.split(/\r?\n/).length },
          { sessionId: orchestrationSessionId, phase },
        )
      }
      lastPlan = planText
    }

    activePhase = null

    const afterSnapshot = await scanSnapshot(project.root)
    const { created, modified, deleted } = diffSnapshots(
      project,
      beforeSnapshot,
      afterSnapshot,
    )

    const finalSummary = latestSummary || 'BioPLEASE completed the run.'
    await overwriteSessionSummary(
      project,
      buildSessionSummaryMarkdown({
        sessionId: orchestrationSessionId,
        goal: options.goal,
        latestSummary: finalSummary,
        createdFiles: created,
        modifiedFiles: modified,
        deletedFiles: deleted,
      }),
    )

    await updateRuntimeState(project, state => ({
      ...state,
      activePhase: null,
      activeSessionId: null,
      latestSummary: finalSummary,
      sessions: state.sessions.map(session =>
        session.id === orchestrationSessionId
          ? {
              ...session,
              status: 'completed',
              completedAt: new Date().toISOString(),
              latestSummary: finalSummary,
              phases: state.phaseStates,
            }
          : session,
      ),
    }))

    await appendJournalEvent(
      project,
      'run_completed',
      {
        goal: options.goal,
        latestSummary: finalSummary,
        createdFiles: created,
        modifiedFiles: modified,
        deletedFiles: deleted,
      },
      { sessionId: orchestrationSessionId },
    )
    await emitEvent(
      project,
      options.onEvent,
      'run_completed',
      finalSummary,
      {
        createdFiles: created,
        modifiedFiles: modified,
        deletedFiles: deleted,
      },
      { sessionId: orchestrationSessionId },
    )

    const finalState = await readRuntimeState(project)
    return {
      sessionId: orchestrationSessionId,
      projectRoot: project.root,
      transcriptPath,
      latestSummary: finalSummary,
      phaseStates: finalState.phaseStates,
      createdFiles: created,
      modifiedFiles: modified,
      deletedFiles: deleted,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    const phaseError = error instanceof BioPleasePhaseError ? error : null
    if (activePhase) {
      await markPhaseState(project, orchestrationSessionId, {
        phase: activePhase,
        status: message === 'Run cancelled' ? 'cancelled' : 'failed',
        model:
          activeModel ??
          (activePhase === 'Plan'
            ? project.config.phaseModelRouting.Plan
            : activePhase === 'Learn'
              ? project.config.phaseModelRouting.Learn
              : activePhase === 'Execute'
                ? project.config.phaseModelRouting.Execute
                : activePhase === 'Assess'
                  ? project.config.phaseModelRouting.Assess
                  : project.config.phaseModelRouting.Share),
        completedAt: new Date().toISOString(),
        summary: latestSummary || message,
        error: message,
        sessionId: phaseError?.openClaudeSessionId ?? undefined,
      })

      await appendJournalEvent(
        project,
        'phase_failed',
        {
          error: message,
          model: activeModel,
          exitCode: phaseError?.exitCode ?? null,
          errorKind: phaseError?.errorKind ?? null,
          openClaudeSessionId: phaseError?.openClaudeSessionId ?? null,
          stopReason: phaseError?.stopReason ?? null,
          numTurns: phaseError?.numTurns ?? null,
          resultErrors: phaseError?.resultErrors ?? [],
          permissionDeniedCount: phaseError?.permissionDenials.length ?? 0,
          permissionDeniedTools:
            phaseError?.permissionDenials.map(record => record.toolName) ?? [],
          stderrTail: phaseError?.stderrTail ?? [],
        },
        { sessionId: orchestrationSessionId, phase: activePhase },
      )
      await emitEvent(
        project,
        options.onEvent,
        'phase_failed',
        message,
        {
          error: message,
          model: activeModel,
          exitCode: phaseError?.exitCode ?? null,
          errorKind: phaseError?.errorKind ?? null,
          openClaudeSessionId: phaseError?.openClaudeSessionId ?? null,
          stopReason: phaseError?.stopReason ?? null,
          numTurns: phaseError?.numTurns ?? null,
          resultErrors: phaseError?.resultErrors ?? [],
          permissionDeniedCount: phaseError?.permissionDenials.length ?? 0,
          permissionDeniedTools:
            phaseError?.permissionDenials.map(record => record.toolName) ?? [],
          stderrTail: phaseError?.stderrTail ?? [],
        },
        { sessionId: orchestrationSessionId, phase: activePhase },
      )
    }
    await updateRuntimeState(project, state => ({
      ...state,
      activePhase: null,
      activeSessionId: null,
      latestSummary: latestSummary || message,
      blockedReason: message,
      sessions: state.sessions.map(session =>
        session.id === orchestrationSessionId
          ? {
              ...session,
              status: message === 'Run cancelled' ? 'cancelled' : 'failed',
              completedAt: new Date().toISOString(),
              error: message,
              latestSummary: latestSummary || state.latestSummary,
            }
          : session,
      ),
    }))

    await appendJournalEvent(
      project,
      message === 'Run cancelled' ? 'run_cancelled' : 'run_failed',
      { error: message },
      { sessionId: orchestrationSessionId },
    )
    await emitEvent(
      project,
      options.onEvent,
      message === 'Run cancelled' ? 'run_cancelled' : 'run_failed',
      message,
      {
        error: message,
        phase: activePhase,
        model: activeModel,
        exitCode: phaseError?.exitCode ?? null,
        errorKind: phaseError?.errorKind ?? null,
        openClaudeSessionId: phaseError?.openClaudeSessionId ?? null,
        stopReason: phaseError?.stopReason ?? null,
        numTurns: phaseError?.numTurns ?? null,
        resultErrors: phaseError?.resultErrors ?? [],
        permissionDeniedCount: phaseError?.permissionDenials.length ?? 0,
        permissionDeniedTools:
          phaseError?.permissionDenials.map(record => record.toolName) ?? [],
      },
      { sessionId: orchestrationSessionId },
    )
    throw error
  }
}
