import { readFile, stat, writeFile } from 'fs/promises'
import { join, resolve } from 'path'
import { fileURLToPath } from 'url'
import { randomUUID } from 'crypto'

import { revokeTool } from '../biocontext.js'
import { formatDoctorReport, runDoctor } from '../doctor.js'
import {
  appendPlanHistory,
  readEnabledTools,
  readRuntimeState,
  readSessionSummary,
} from '../journal.js'
import { listRecentProjects, openProjectFolder, getProjectSnapshot, loadProjectByRoot } from '../projects.js'
import { runBioPleaseSession, type RunSummary } from '../runner.js'
import type { BioPleaseEvent } from '../types.js'
import { execFileNoThrow } from '../../utils/execFileNoThrow.js'

type JobStatus = 'running' | 'completed' | 'failed' | 'cancelled'

type WebJob = {
  id: string
  projectRoot: string
  goal: string
  status: JobStatus
  createdAt: string
  updatedAt: string
  summary?: RunSummary
  error?: string
  abortController: AbortController
}

type ProjectEventBuffer = {
  events: BioPleaseEvent[]
  streams: Set<ReadableStreamDefaultController<string>>
}

const WEB_ROOT = fileURLToPath(new URL('./', import.meta.url))
const jobs = new Map<string, WebJob>()
const eventBuffers = new Map<string, ProjectEventBuffer>()

function json(data: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(data, null, 2), {
    ...init,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...(init?.headers ?? {}),
    },
  })
}

function text(body: string, init?: ResponseInit): Response {
  return new Response(body, init)
}

async function parseJsonBody(request: Request): Promise<any> {
  try {
    return await request.json()
  } catch {
    return null
  }
}

function getProjectRoot(request: Request): string | null {
  const url = new URL(request.url)
  return url.searchParams.get('root')
}

function getEventBuffer(projectRoot: string): ProjectEventBuffer {
  if (!eventBuffers.has(projectRoot)) {
    eventBuffers.set(projectRoot, { events: [], streams: new Set() })
  }
  return eventBuffers.get(projectRoot)!
}

async function publishEvent(event: BioPleaseEvent): Promise<void> {
  const buffer = getEventBuffer(event.projectRoot)
  buffer.events.push(event)
  if (buffer.events.length > 800) {
    buffer.events.splice(0, buffer.events.length - 800)
  }
  const payload = `data: ${JSON.stringify(event)}\n\n`
  for (const stream of buffer.streams) {
    try {
      stream.enqueue(payload)
    } catch {}
  }
}

function getStaticAsset(pathname: string): string | null {
  if (pathname.startsWith('/assets/') && !pathname.includes('..')) {
    return pathname.slice(1)
  }

  const map: Record<string, string> = {
    '/': 'index.html',
    '/index.html': 'index.html',
    '/app.js': 'app.js',
    '/styles.css': 'styles.css',
  }
  return map[pathname] ?? null
}

async function serveStatic(pathname: string): Promise<Response | null> {
  const file = getStaticAsset(pathname)
  if (!file) {
    return null
  }

  const response = new Response(Bun.file(join(WEB_ROOT, file)))
  response.headers.set('cache-control', 'no-store')
  if (file.endsWith('.html')) {
    response.headers.set('content-type', 'text/html; charset=utf-8')
  }
  if (file.endsWith('.js')) {
    response.headers.set('content-type', 'text/javascript; charset=utf-8')
  }
  if (file.endsWith('.css')) {
    response.headers.set('content-type', 'text/css; charset=utf-8')
  }
  if (file.endsWith('.svg')) {
    response.headers.set('content-type', 'image/svg+xml')
  }
  return response
}

async function handleProjectOpen(request: Request): Promise<Response> {
  const body = await parseJsonBody(request)
  if (!body?.projectRoot || typeof body.projectRoot !== 'string') {
    return json({ error: 'projectRoot is required' }, { status: 400 })
  }

  const project = await openProjectFolder({
    projectRoot: body.projectRoot,
    title: typeof body.title === 'string' ? body.title : undefined,
    researchQuestion:
      typeof body.researchQuestion === 'string' ? body.researchQuestion : undefined,
    backgroundContext:
      typeof body.backgroundContext === 'string' ? body.backgroundContext : undefined,
  })

  return json({
    project: {
      root: project.root,
      config: project.config,
    },
  })
}

async function handleProjectReveal(request: Request): Promise<Response> {
  const body = await parseJsonBody(request)
  if (!body?.root || typeof body.root !== 'string') {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const projectRoot = resolve(body.root)
  const projectStat = await stat(projectRoot).catch(() => null)
  if (!projectStat?.isDirectory()) {
    return json({ error: 'Project folder not found.' }, { status: 404 })
  }

  const command =
    process.platform === 'win32'
      ? { file: 'cmd', args: ['/c', 'start', '', projectRoot] }
      : process.platform === 'darwin'
        ? { file: 'open', args: [projectRoot] }
        : { file: 'xdg-open', args: [projectRoot] }

  const result = await execFileNoThrow(command.file, command.args, {
    useCwd: false,
    timeout: 5000,
  })

  if (result.code !== 0) {
    return json(
      {
        error:
          result.error || result.stderr.trim() || `Failed to open folder (exit ${result.code}).`,
      },
      { status: 500 },
    )
  }

  return json({ ok: true, revealedPath: projectRoot })
}

async function handleProjectPickFolder(): Promise<Response> {
  if (process.platform !== 'win32') {
    return json({ error: 'Native folder picking is only supported on Windows in the web app.' }, { status: 501 })
  }

  const pickerScript =
    "Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.FolderBrowserDialog; $dialog.Description = 'Select the main BioPLEASE folder'; $dialog.ShowNewFolderButton = $true; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dialog.SelectedPath }"

  const result = await execFileNoThrow(
    'powershell',
    ['-NoProfile', '-STA', '-Command', pickerScript],
    {
      useCwd: false,
      timeout: 60 * 60 * 1000,
    },
  )

  if (result.code !== 0) {
    return json(
      {
        error:
          result.error || result.stderr.trim() || `Folder picker failed (exit ${result.code}).`,
      },
      { status: 500 },
    )
  }

  const pickedRoot =
    result.stdout
      .split(/\r?\n/)
      .map(line => line.trim())
      .find(Boolean) || null

  return json({
    cancelled: !pickedRoot,
    root: pickedRoot,
  })
}

async function handleProjectSnapshot(request: Request): Promise<Response> {
  const projectRoot = getProjectRoot(request)
  if (!projectRoot) {
    return json({ error: 'root is required' }, { status: 400 })
  }

  try {
    const snapshot = await getProjectSnapshot(projectRoot)
    return json({ snapshot })
  } catch (error) {
    return json(
      { error: error instanceof Error ? error.message : 'failed to load project' },
      { status: 400 },
    )
  }
}

async function handlePlanRead(request: Request): Promise<Response> {
  const projectRoot = getProjectRoot(request)
  if (!projectRoot) {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const project = await loadProjectByRoot(projectRoot)
  const plan = await readFile(project.paths.plan, 'utf8').catch(() => '')
  return json({ plan })
}

async function handlePlanWrite(request: Request): Promise<Response> {
  const body = await parseJsonBody(request)
  if (!body?.root || typeof body.root !== 'string' || typeof body.plan !== 'string') {
    return json({ error: 'root and plan are required' }, { status: 400 })
  }

  const project = await loadProjectByRoot(body.root)
  await writeFile(project.paths.plan, body.plan, 'utf8')
  await appendPlanHistory(project, body.plan, { actor: 'user' })
  return json({ ok: true })
}

async function handleOutputs(request: Request): Promise<Response> {
  const projectRoot = getProjectRoot(request)
  if (!projectRoot) {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const snapshot = await getProjectSnapshot(projectRoot)
  return json({
    outputs: snapshot.artifacts.slice(0, 50),
    summary: snapshot.summary,
  })
}

async function handleTools(request: Request): Promise<Response> {
  const projectRoot = getProjectRoot(request)
  if (!projectRoot) {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const project = await loadProjectByRoot(projectRoot)
  const tools = await readEnabledTools(project)
  return json({ tools })
}

async function handleToolRevoke(request: Request): Promise<Response> {
  const body = await parseJsonBody(request)
  if (!body?.root || !body?.toolId) {
    return json({ error: 'root and toolId are required' }, { status: 400 })
  }

  const project = await loadProjectByRoot(body.root)
  const tools = await revokeTool(project, String(body.toolId))
  return json({ tools })
}

async function handleRunStart(request: Request): Promise<Response> {
  const body = await parseJsonBody(request)
  if (!body?.root || typeof body.root !== 'string' || typeof body.goal !== 'string') {
    return json({ error: 'root and goal are required' }, { status: 400 })
  }

  if (jobs.has(body.root)) {
    return json({ error: 'A run is already active for this project.' }, { status: 409 })
  }

  const job: WebJob = {
    id: randomUUID(),
    projectRoot: body.root,
    goal: body.goal,
    status: 'running',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    abortController: new AbortController(),
  }
  jobs.set(body.root, job)

  void (async () => {
    try {
      const summary = await runBioPleaseSession({
        projectRoot: body.root,
        goal: body.goal,
        maxTurns:
          typeof body.maxTurns === 'number' ? body.maxTurns : undefined,
        permissionMode:
          typeof body.permissionMode === 'string' ? body.permissionMode : undefined,
        signal: job.abortController.signal,
        onEvent: publishEvent,
      })
      job.status = 'completed'
      job.summary = summary
    } catch (error) {
      job.status =
        job.abortController.signal.aborted ? 'cancelled' : 'failed'
      job.error = error instanceof Error ? error.message : String(error)
    } finally {
      job.updatedAt = new Date().toISOString()
      jobs.delete(body.root)
    }
  })()

  return json({ job }, { status: 202 })
}

async function handleRunCancel(request: Request): Promise<Response> {
  const body = await parseJsonBody(request)
  if (!body?.root || typeof body.root !== 'string') {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const job = jobs.get(body.root)
  if (!job) {
    return json({ error: 'No active run for this project.' }, { status: 404 })
  }

  job.abortController.abort()
  job.status = 'cancelled'
  job.updatedAt = new Date().toISOString()
  return json({ ok: true })
}

async function handleRunStatus(request: Request): Promise<Response> {
  const projectRoot = getProjectRoot(request)
  if (!projectRoot) {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const job = jobs.get(projectRoot) ?? null
  const project = await loadProjectByRoot(projectRoot)
  const [state, summary] = await Promise.all([
    readRuntimeState(project),
    readSessionSummary(project),
  ])
  return json({ job, state, summary })
}

async function handleEvents(request: Request): Promise<Response> {
  const projectRoot = getProjectRoot(request)
  if (!projectRoot) {
    return json({ error: 'root is required' }, { status: 400 })
  }

  const buffer = getEventBuffer(projectRoot)
  let localController: ReadableStreamDefaultController<string> | null = null

  const stream = new ReadableStream<string>({
    start(controller) {
      localController = controller
      buffer.streams.add(controller)
      controller.enqueue(`event: ready\ndata: {"ok":true}\n\n`)
      for (const event of buffer.events.slice(-250)) {
        controller.enqueue(`data: ${JSON.stringify(event)}\n\n`)
      }
    },
    cancel() {
      if (localController) {
        buffer.streams.delete(localController)
      }
    },
  })

  return new Response(stream, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
    },
  })
}

async function handleApi(request: Request, pathname: string): Promise<Response> {
  if (pathname === '/api/doctor' && request.method === 'GET') {
    return json({ report: runDoctor(), formatted: formatDoctorReport(runDoctor()) })
  }

  if (pathname === '/api/projects/recent' && request.method === 'GET') {
    return json({ projects: await listRecentProjects() })
  }

  if (pathname === '/api/projects/open' && request.method === 'POST') {
    return handleProjectOpen(request)
  }

  if (pathname === '/api/project/reveal' && request.method === 'POST') {
    return handleProjectReveal(request)
  }

  if (pathname === '/api/project/pick-folder' && request.method === 'POST') {
    return handleProjectPickFolder()
  }

  if (pathname === '/api/project' && request.method === 'GET') {
    return handleProjectSnapshot(request)
  }

  if (pathname === '/api/project/plan' && request.method === 'GET') {
    return handlePlanRead(request)
  }

  if (pathname === '/api/project/plan' && request.method === 'PUT') {
    return handlePlanWrite(request)
  }

  if (pathname === '/api/project/outputs' && request.method === 'GET') {
    return handleOutputs(request)
  }

  if (pathname === '/api/project/tools' && request.method === 'GET') {
    return handleTools(request)
  }

  if (pathname === '/api/project/tools/revoke' && request.method === 'POST') {
    return handleToolRevoke(request)
  }

  if (pathname === '/api/project/run' && request.method === 'POST') {
    return handleRunStart(request)
  }

  if (pathname === '/api/project/run' && request.method === 'GET') {
    return handleRunStatus(request)
  }

  if (pathname === '/api/project/cancel' && request.method === 'POST') {
    return handleRunCancel(request)
  }

  if (pathname === '/api/project/events' && request.method === 'GET') {
    return handleEvents(request)
  }

  return json({ error: 'not found' }, { status: 404 })
}

export function startBioPleaseWebServer(options?: { port?: number }) {
  const port = options?.port ?? 4114
  const server = Bun.serve({
    port,
    async fetch(request) {
      const url = new URL(request.url)
      if (url.pathname.startsWith('/api/')) {
        return handleApi(request, url.pathname)
      }

      const staticResponse = await serveStatic(url.pathname)
      if (staticResponse) {
        return staticResponse
      }

      return text('Not found', { status: 404 })
    },
  })

  process.stdout.write(
    `BioPLEASE app running at http://localhost:${server.port}\n`,
  )
  return server
}

if (import.meta.main) {
  const parsedPort = Number.parseInt(process.env.PORT ?? '4114', 10)
  startBioPleaseWebServer({
    port: Number.isNaN(parsedPort) ? 4114 : parsedPort,
  })
}
