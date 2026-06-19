const state = {
  currentRoot: '',
  snapshot: null,
  recentProjects: [],
  transcriptEvents: [],
  activeJob: null,
  eventSource: null,
  refreshTimer: null,
}

const els = {
  projectRoot: document.getElementById('project-root'),
  openProject: document.getElementById('open-project'),
  pickFolder: document.getElementById('pick-folder'),
  refreshProject: document.getElementById('refresh-project'),
  doctor: document.getElementById('doctor'),
  recentProjects: document.getElementById('recent-projects'),
  sessionList: document.getElementById('session-list'),
  projectTitle: document.getElementById('project-title'),
  projectQuestion: document.getElementById('project-question'),
  runIndicator: document.getElementById('run-indicator'),
  cancelRun: document.getElementById('cancel-run'),
  transcript: document.getElementById('transcript'),
  composer: document.getElementById('composer'),
  goalInput: document.getElementById('goal-input'),
  maxTurns: document.getElementById('max-turns'),
  currentRootLabel: document.getElementById('current-root-label'),
  latestSummaryChip: document.getElementById('latest-summary-chip'),
  planEditor: document.getElementById('plan-editor'),
  savePlan: document.getElementById('save-plan'),
  phaseProgress: document.getElementById('phase-progress'),
  recentSummary: document.getElementById('recent-summary'),
  outputsList: document.getElementById('outputs-list'),
  toolsList: document.getElementById('tools-list'),
  startRun: document.getElementById('start-run'),
}

const projectsHomeRoot = 'G:\\BioClaw\\projects'

function normalizeProjectRoot(root) {
  return typeof root === 'string' ? root.trim() : ''
}

function setProjectRootDisplay(root) {
  const normalizedRoot = normalizeProjectRoot(root)
  const displayRoot = normalizedRoot || 'No working folder selected'
  els.projectRoot.textContent = displayRoot
  els.projectRoot.title = displayRoot
  els.projectRoot.dataset.root = normalizedRoot
}

function getSelectedProjectRoot() {
  return normalizeProjectRoot(state.currentRoot || els.projectRoot.dataset.root || '')
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

async function api(path, options) {
  const response = await fetch(path, {
    headers: {
      'content-type': 'application/json',
      ...(options?.headers ?? {}),
    },
    ...options,
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`)
  }
  return data
}

function setIndicator(status, label = status) {
  els.runIndicator.className = `status-pill ${status}`
  els.runIndicator.textContent = label
}

function formatDate(value) {
  if (!value) {
    return 'Unknown time'
  }
  return new Date(value).toLocaleString()
}

function formatMultilineText(value) {
  return escapeHtml(String(value ?? '')).replaceAll('\n', '<br />')
}

function formatDuration(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return null
  }

  if (value < 1000) {
    return `${Math.round(value)} ms`
  }

  const seconds = value / 1000
  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`
  }

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.round(seconds % 60)
  return `${minutes}m ${remainingSeconds}s`
}

function formatElapsedSeconds(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return null
  }

  if (value < 60) {
    return `${Math.round(value)} s`
  }

  const minutes = Math.floor(value / 60)
  const seconds = Math.round(value % 60)
  return `${minutes}m ${seconds}s`
}

function formatCompactNumber(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }
  return new Intl.NumberFormat(undefined, {
    notation: value >= 1000 ? 'compact' : 'standard',
    maximumFractionDigits: value >= 1000 ? 1 : 0,
  }).format(value)
}

function formatCurrency(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }
  return `$${value.toFixed(value >= 1 ? 2 : 4)}`
}

function getEventData(event) {
  return event?.data && typeof event.data === 'object' ? event.data : {}
}

function getEventRole(event) {
  if (event.type === 'run_started') {
    return 'user'
  }
  return 'agent'
}

function getEventMeta(event) {
  const data = getEventData(event)
  const labels = []

  if (event.phase) {
    labels.push(event.phase)
  }

  if (event.type === 'phase_output') {
    labels.push(
      typeof data.outputKind === 'string'
        ? data.outputKind.replaceAll('_', ' ')
        : 'openclaude',
    )
  } else {
    labels.push(event.type.replaceAll('_', ' '))
  }

  return labels.filter(Boolean).join(' | ')
}

function buildEventDetailLines(event) {
  const data = getEventData(event)
  const lines = []
  const usage =
    data.usage && typeof data.usage === 'object' ? data.usage : null

  if (event.type === 'phase_started' && typeof data.model === 'string') {
    lines.push(`Model: ${data.model}`)
  }

  if (event.type === 'phase_completed') {
    if (typeof data.model === 'string') {
      lines.push(`Model: ${data.model}`)
    }
    if (typeof data.openClaudeSessionId === 'string' && data.openClaudeSessionId) {
      lines.push(`OpenClaude session: ${data.openClaudeSessionId}`)
    }
  }

  if (event.type === 'phase_failed' || event.type === 'run_failed') {
    if (typeof data.model === 'string' && data.model) {
      lines.push(`Model: ${data.model}`)
    }
    if (typeof data.exitCode === 'number') {
      lines.push(`Exit code: ${data.exitCode}`)
    }
    if (typeof data.errorKind === 'string' && data.errorKind) {
      lines.push(`Failure type: ${data.errorKind}`)
    }
    if (typeof data.stopReason === 'string' && data.stopReason) {
      lines.push(`Stop reason: ${data.stopReason}`)
    }
    if (typeof data.numTurns === 'number') {
      lines.push(`Turns: ${data.numTurns}`)
    }
    if (typeof data.permissionDeniedCount === 'number' && data.permissionDeniedCount > 0) {
      lines.push(`Permission denials: ${data.permissionDeniedCount}`)
    }
    if (Array.isArray(data.permissionDeniedTools) && data.permissionDeniedTools.length > 0) {
      lines.push(`Denied tools: ${data.permissionDeniedTools.join(', ')}`)
    }
    if (Array.isArray(data.resultErrors) && data.resultErrors.length > 0) {
      lines.push(`Errors: ${data.resultErrors.join(' | ')}`)
    }
    if (typeof data.openClaudeSessionId === 'string' && data.openClaudeSessionId) {
      lines.push(`OpenClaude session: ${data.openClaudeSessionId}`)
    }
  }

  if (event.type === 'tool_started') {
    if (typeof data.toolName === 'string' && data.toolName) {
      lines.push(`Tool: ${data.toolName}`)
    }
    if (typeof data.inputSummary === 'string' && data.inputSummary) {
      lines.push(`Input: ${data.inputSummary}`)
    }
    if (typeof data.taskType === 'string' && data.taskType) {
      lines.push(`Task type: ${data.taskType}`)
    }
    if (typeof data.workflowName === 'string' && data.workflowName) {
      lines.push(`Workflow: ${data.workflowName}`)
    }
    if (typeof data.toolUseId === 'string' && data.toolUseId) {
      lines.push(`Tool use id: ${data.toolUseId}`)
    }
  }

  if (event.type === 'tool_progress') {
    if (typeof data.toolName === 'string' && data.toolName) {
      lines.push(`Tool: ${data.toolName}`)
    }
    const elapsed = formatElapsedSeconds(data.elapsedTimeSeconds)
    if (elapsed) {
      lines.push(`Elapsed: ${elapsed}`)
    }
    if (typeof data.taskId === 'string' && data.taskId) {
      lines.push(`Task id: ${data.taskId}`)
    }
  }

  if (event.type === 'task_progress') {
    if (typeof data.lastToolName === 'string' && data.lastToolName) {
      lines.push(`Latest tool: ${data.lastToolName}`)
    }
    if (
      typeof data.summary === 'string' &&
      data.summary &&
      data.summary !== event.message
    ) {
      lines.push(`Summary: ${data.summary}`)
    }
  }

  if (event.type === 'tool_finished') {
    if (typeof data.toolName === 'string' && data.toolName) {
      lines.push(`Tool: ${data.toolName}`)
    }
    if (typeof data.inputSummary === 'string' && data.inputSummary) {
      lines.push(`Input: ${data.inputSummary}`)
    }
    if (typeof data.status === 'string' && data.status) {
      lines.push(`Status: ${data.status}`)
    }
    if (typeof data.resultPreview === 'string' && data.resultPreview) {
      lines.push(`Result: ${data.resultPreview}`)
    }
    if (typeof data.outputFile === 'string' && data.outputFile) {
      lines.push(`Output file: ${data.outputFile}`)
    }
  }

  if (usage) {
    const totalTokens = formatCompactNumber(usage.total_tokens)
    const toolUses = formatCompactNumber(usage.tool_uses)
    const duration = formatDuration(usage.duration_ms)

    if (totalTokens) {
      lines.push(`Tokens: ${totalTokens}`)
    }
    if (toolUses) {
      lines.push(`Tool uses: ${toolUses}`)
    }
    if (duration) {
      lines.push(`Duration: ${duration}`)
    }
  }

  if (event.type === 'phase_output') {
    if (data.outputKind === 'init') {
      if (typeof data.model === 'string' && data.model) {
        lines.push(`Model: ${data.model}`)
      }
      if (typeof data.cwd === 'string' && data.cwd) {
        lines.push(`Working directory: ${data.cwd}`)
      }
      if (typeof data.permissionMode === 'string' && data.permissionMode) {
        lines.push(`Permission mode: ${data.permissionMode}`)
      }
      if (typeof data.toolsCount === 'number') {
        lines.push(`Available tools: ${data.toolsCount}`)
      }
      if (Array.isArray(data.toolsPreview) && data.toolsPreview.length > 0) {
        lines.push(`Tool preview: ${data.toolsPreview.join(', ')}`)
      }
      if (Array.isArray(data.mcpServers) && data.mcpServers.length > 0) {
        lines.push(`MCP servers: ${data.mcpServers.join(', ')}`)
      }
      if (Array.isArray(data.skills) && data.skills.length > 0) {
        lines.push(`Skills: ${data.skills.join(', ')}`)
      }
      if (Array.isArray(data.slashCommands) && data.slashCommands.length > 0) {
        lines.push(`Slash commands: ${data.slashCommands.join(', ')}`)
      }
    }

    if (data.outputKind === 'api_retry') {
      if (
        typeof data.attempt === 'number' &&
        typeof data.maxRetries === 'number'
      ) {
        lines.push(`Retry: ${data.attempt}/${data.maxRetries}`)
      }
      if (typeof data.errorStatus === 'number') {
        lines.push(`Status: ${data.errorStatus}`)
      }
      if (typeof data.error === 'string' && data.error) {
        lines.push(`Reason: ${data.error}`)
      }
      const backoff = formatDuration(data.retryDelayMs)
      if (backoff) {
        lines.push(`Backoff: ${backoff}`)
      }
    }

    if (data.outputKind === 'result') {
      if (typeof data.numTurns === 'number') {
        lines.push(`Turns: ${data.numTurns}`)
      }
      const duration = formatDuration(data.durationMs)
      if (duration) {
        lines.push(`Duration: ${duration}`)
      }
      const totalCost = formatCurrency(data.totalCostUsd)
      if (totalCost) {
        lines.push(`Cost: ${totalCost}`)
      }
      if (typeof data.stopReason === 'string' && data.stopReason) {
        lines.push(`Stop reason: ${data.stopReason}`)
      }
      if (Array.isArray(data.errors) && data.errors.length > 0) {
        lines.push(`Errors: ${data.errors.join(' | ')}`)
      }
      if (typeof data.permissionDeniedCount === 'number' && data.permissionDeniedCount > 0) {
        lines.push(`Permission denials: ${data.permissionDeniedCount}`)
      }
      if (Array.isArray(data.permissionDeniedTools) && data.permissionDeniedTools.length > 0) {
        lines.push(`Denied tools: ${data.permissionDeniedTools.join(', ')}`)
      }
    }

    if (data.outputKind === 'stderr') {
      lines.push('Runtime stderr')
    }
  }

  if (event.type === 'tool_enabled' && typeof data.url === 'string' && data.url) {
    lines.push(`Source: ${data.url}`)
  }

  return lines
}

function renderEventDetails(event) {
  const detailLines = buildEventDetailLines(event)
  if (detailLines.length === 0) {
    return ''
  }

  return `
    <div class="message-detail">
      ${detailLines
        .map(line => `<div>${formatMultilineText(line)}</div>`)
        .join('')}
    </div>
  `
}

function getRecentPhaseActivity(phaseName) {
  const recentEvent = [...state.transcriptEvents]
    .reverse()
    .find(
      event =>
        event.phase === phaseName &&
        ['phase_output', 'tool_started', 'tool_progress', 'task_progress', 'tool_finished'].includes(
          event.type,
        ) &&
        typeof event.message === 'string' &&
        event.message.trim(),
    )

  return recentEvent?.message?.trim() || ''
}

function getRecentPhaseActivities(phaseName, limit = 3) {
  const seen = new Set()

  return [...state.transcriptEvents]
    .reverse()
    .filter(
      event =>
        event.phase === phaseName &&
        ['phase_output', 'tool_started', 'tool_progress', 'task_progress', 'tool_finished'].includes(
          event.type,
        ) &&
        typeof event.message === 'string' &&
        event.message.trim(),
    )
    .map(event => event.message.trim())
    .filter(message => {
      if (seen.has(message)) {
        return false
      }
      seen.add(message)
      return true
    })
    .slice(0, limit)
}

function pushTranscriptEvent(event) {
  state.transcriptEvents.push(event)
  if (state.transcriptEvents.length > 250) {
    state.transcriptEvents.splice(0, state.transcriptEvents.length - 250)
  }
  renderTranscript()
}

function notify(message, type = 'summary_updated') {
  pushTranscriptEvent({
    type,
    message,
  })
}

async function runUiAction(action, failureLabel = 'Action failed') {
  try {
    await action()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    notify(message, 'run_failed')
    setIndicator('failed', failureLabel)
  }
}

function scheduleRefresh(delay = 400) {
  if (!state.currentRoot) {
    return
  }
  clearTimeout(state.refreshTimer)
  state.refreshTimer = setTimeout(() => {
    void refreshProject()
  }, delay)
}

function renderRecentProjects() {
  if (state.recentProjects.length === 0) {
    els.recentProjects.className = 'recent-list empty-state'
    els.recentProjects.textContent =
      'Open a local folder to start tracking projects here.'
    return
  }

  els.recentProjects.className = 'recent-list'
  els.recentProjects.innerHTML = state.recentProjects
    .map(
      project => `
        <button class="recent-item" type="button" data-root="${escapeHtml(project.root)}">
          <strong>${escapeHtml(project.title)}</strong>
          <span class="meta-line">${escapeHtml(project.root)}</span>
          <span class="meta-line">${escapeHtml(project.question || 'No question recorded')}</span>
          <span class="meta-line">${escapeHtml(formatDate(project.updatedAt))}</span>
        </button>
      `,
    )
    .join('')
}

function renderSessions() {
  const sessions = state.snapshot?.state?.sessions ?? []
  if (sessions.length === 0) {
    els.sessionList.className = 'session-list empty-state'
    els.sessionList.textContent = 'No session history yet.'
    return
  }

  els.sessionList.className = 'session-list'
  els.sessionList.innerHTML = sessions
    .slice(0, 8)
    .map(
      session => `
        <div class="session-item">
          <strong>${escapeHtml(session.goal)}</strong>
          <span class="meta-line">${escapeHtml(session.status)}</span>
          <span class="meta-line">${escapeHtml(formatDate(session.startedAt))}</span>
          <span class="meta-line">${escapeHtml(session.latestSummary || 'No summary')}</span>
        </div>
      `,
    )
    .join('')
}

function renderProjectHeader() {
  const config = state.snapshot?.project?.config
  if (!config) {
    els.projectTitle.textContent = 'No project selected'
    els.projectQuestion.textContent =
      'Open a folder to start a BioPLEASE run.'
    els.currentRootLabel.textContent = 'No working folder selected'
    els.latestSummaryChip.textContent = 'No summary yet'
    return
  }

  els.projectTitle.textContent = config.title
  els.projectQuestion.textContent =
    config.researchQuestion || 'No research question recorded.'
  els.currentRootLabel.textContent = state.currentRoot
  els.latestSummaryChip.textContent =
    state.snapshot.summary?.replace(/^#.*$/gm, '').trim().slice(0, 120) ||
    state.snapshot.state?.latestSummary ||
    'No summary yet'
}

function renderTranscript() {
  const events = state.transcriptEvents
  els.transcript.innerHTML = [
    `
    <div class="message message-system">
      <div class="message-meta">BioPLEASE</div>
      <div class="message-body">
        This local app runs OpenClaude directly inside your chosen folder, keeps runtime state in
        <code>.bioplease/</code>, and shows the live plan, progress, summary, and outputs as the agent works.
      </div>
    </div>
    `,
    ...events.map(event => {
      const role = getEventRole(event)
      const meta = getEventMeta(event)
      return `
        <div class="message ${role} event-${escapeHtml(event.type)}">
          <div class="message-meta">${escapeHtml(meta || 'event')}</div>
          <div class="message-body">
            <div class="message-copy">${formatMultilineText(event.message || '')}</div>
            ${renderEventDetails(event)}
          </div>
        </div>
      `
    }),
  ].join('')
  els.transcript.scrollTop = els.transcript.scrollHeight
}

function renderPlan() {
  els.planEditor.value = state.snapshot?.plan || ''
}

function renderProgress() {
  const phaseStates = state.snapshot?.state?.phaseStates ?? []

  if (phaseStates.length === 0) {
    els.phaseProgress.className = 'phase-progress empty-state'
    els.phaseProgress.textContent = 'No progress yet.'
  } else {
    els.phaseProgress.className = 'phase-progress'
    els.phaseProgress.innerHTML = phaseStates
      .map(phase => {
        return `
          <div class="phase-row">
            <div class="phase-main">
              <strong>${escapeHtml(phase.phase)}</strong>
              <span>${escapeHtml(phase.model)}</span>
              ${
                phase.error || phase.summary
                  ? `<span class="phase-note">${escapeHtml(
                      phase.error || phase.summary,
                    )}</span>`
                  : ''
              }
            </div>
            <div class="phase-tag ${escapeHtml(phase.status)}">${escapeHtml(phase.status)}</div>
          </div>
        `
      })
      .join('')
  }
}

function renderSummary() {
  const summary =
    state.snapshot?.summary ||
    state.snapshot?.state?.latestSummary ||
    state.snapshot?.state?.blockedReason ||
    'No summary yet.'
  els.recentSummary.className = 'summary-card'
  els.recentSummary.textContent = summary
}

function renderOutputs() {
  const outputs =
    (state.snapshot?.artifacts ?? []).filter(
      output =>
        output.relativePath.startsWith('reports/') ||
        output.relativePath.startsWith('artifacts/') ||
        output.relativePath.startsWith('figures/') ||
        output.relativePath.startsWith('data/') ||
        output.relativePath.endsWith('.html'),
    ) || []
  if (outputs.length === 0) {
    els.outputsList.className = 'outputs-list empty-state'
    els.outputsList.textContent = 'No artifacts yet.'
    return
  }

  els.outputsList.className = 'outputs-list'
  els.outputsList.innerHTML = outputs
    .slice(0, 20)
    .map(
      output => `
        <div class="output-item">
          <strong>${escapeHtml(output.relativePath)}</strong>
          <span>${escapeHtml(output.kind)} | ${escapeHtml(String(output.sizeBytes))} bytes</span>
          <span>${escapeHtml(formatDate(output.modifiedAt))}</span>
        </div>
      `,
    )
    .join('')
}

function renderTools() {
  const tools = state.snapshot?.enabledTools ?? []
  if (tools.length === 0) {
    els.toolsList.className = 'tools-list empty-state'
    els.toolsList.textContent = 'BioContext tools will appear here.'
    return
  }

  els.toolsList.className = 'tools-list'
  els.toolsList.innerHTML = tools
    .map(
      tool => `
        <div class="tool-item">
          <strong>${escapeHtml(tool.name)}</strong>
          <span>${escapeHtml(tool.description)}</span>
          <span>${escapeHtml(tool.url || tool.provenance || '')}</span>
          <button class="button ghost tiny tool-revoke" data-tool-id="${escapeHtml(tool.id)}">Revoke</button>
        </div>
      `,
    )
    .join('')
}

function renderAll() {
  renderRecentProjects()
  renderSessions()
  renderProjectHeader()
  renderPlan()
  renderProgress()
  renderSummary()
  renderOutputs()
  renderTools()
}

async function loadRecentProjects() {
  const data = await api('/api/projects/recent')
  state.recentProjects = data.projects || []
  renderRecentProjects()
}

async function refreshProject() {
  if (!state.currentRoot) {
    return
  }

  const snapshotData = await api(
    `/api/project?root=${encodeURIComponent(state.currentRoot)}`,
  )
  state.snapshot = snapshotData.snapshot

  const runData = await api(
    `/api/project/run?root=${encodeURIComponent(state.currentRoot)}`,
  )
  state.activeJob = runData.job
  setIndicator(state.activeJob?.status || 'idle', state.activeJob?.status || 'Idle')
  els.cancelRun.disabled = !state.activeJob

  renderAll()
}

function connectEvents(root) {
  if (state.eventSource) {
    state.eventSource.close()
  }

  state.transcriptEvents = []
  renderTranscript()

  const source = new EventSource(
    `/api/project/events?root=${encodeURIComponent(root)}`,
  )
  source.onmessage = event => {
    const payload = JSON.parse(event.data)
    pushTranscriptEvent(payload)
    if (
      [
        'phase_started',
        'phase_completed',
        'phase_failed',
        'plan_updated',
        'summary_updated',
        'run_completed',
        'run_failed',
        'run_cancelled',
        'tool_enabled',
        'tool_revoked',
      ].includes(payload.type)
    ) {
      scheduleRefresh()
    }
  }
  state.eventSource = source
}

async function openProject(root, extra = {}) {
  const projectRoot = normalizeProjectRoot(root) || getSelectedProjectRoot()
  if (!projectRoot) {
    notify('Pick a project folder first.', 'run_failed')
    return
  }

  await api('/api/projects/open', {
    method: 'POST',
    body: JSON.stringify({
      projectRoot,
      ...extra,
    }),
  })

  state.currentRoot = projectRoot
  setProjectRootDisplay(projectRoot)
  connectEvents(projectRoot)
  await Promise.all([refreshProject(), loadRecentProjects()])
}

async function revealProjectInExplorer(root) {
  const projectRoot = normalizeProjectRoot(root) || getSelectedProjectRoot()
  if (!projectRoot) {
    notify('Pick a project folder first.', 'run_failed')
    return
  }

  const data = await api('/api/project/reveal', {
    method: 'POST',
    body: JSON.stringify({
      root: projectRoot,
    }),
  })

  notify(`Opened ${data.revealedPath} in File Explorer.`)
}

async function savePlan() {
  if (!state.currentRoot) {
    return
  }

  await api('/api/project/plan', {
    method: 'PUT',
    body: JSON.stringify({
      root: state.currentRoot,
      plan: els.planEditor.value,
    }),
  })
  pushTranscriptEvent({
    type: 'plan_updated',
    message: 'Saved plan edits from the app.',
  })
  await refreshProject()
}

async function startRun(event) {
  event.preventDefault()
  if (!state.currentRoot) {
    return
  }
  const goal = els.goalInput.value.trim()
  if (!goal) {
    return
  }

  pushTranscriptEvent({
    type: 'run_started',
    message: goal,
  })

  await api('/api/project/run', {
    method: 'POST',
    body: JSON.stringify({
      root: state.currentRoot,
      goal,
      maxTurns: Number(els.maxTurns.value || 45),
    }),
  })
  els.goalInput.value = ''
  setIndicator('running', 'running')
  els.cancelRun.disabled = false
  scheduleRefresh(150)
}

async function cancelRun() {
  if (!state.currentRoot) {
    return
  }
  await api('/api/project/cancel', {
    method: 'POST',
    body: JSON.stringify({ root: state.currentRoot }),
  })
  setIndicator('cancelled', 'cancelled')
  els.cancelRun.disabled = true
  scheduleRefresh(150)
}

async function showDoctor() {
  const data = await api('/api/doctor')
  pushTranscriptEvent({
    type: 'summary_updated',
    message: data.formatted,
  })
}

async function pickFolder() {
  if (window.biopleaseDesktop?.pickFolder) {
    const picked = await window.biopleaseDesktop.pickFolder()
    if (picked) {
      await openProject(picked)
    } else {
      notify('Folder selection cancelled.')
    }
    return
  }

  const data = await api('/api/project/pick-folder', {
    method: 'POST',
    body: JSON.stringify({}),
  })

  if (data.cancelled || !data.root) {
    notify('Folder selection cancelled.')
    return
  }

  const picked = String(data.root).trim()
  if (!picked) {
    notify('Folder path cannot be empty.', 'run_failed')
    return
  }

  await openProject(picked)
}

els.openProject.addEventListener('click', () => {
  void runUiAction(
    () => revealProjectInExplorer(getSelectedProjectRoot()),
    'Explorer open failed',
  )
})

els.refreshProject.addEventListener('click', () => {
  void runUiAction(() => refreshProject(), 'Refresh failed')
})

els.pickFolder.addEventListener('click', () => {
  void runUiAction(() => pickFolder(), 'Folder select failed')
})

els.doctor.addEventListener('click', () => {
  void runUiAction(() => showDoctor(), 'Doctor failed')
})

els.savePlan.addEventListener('click', () => {
  void runUiAction(() => savePlan(), 'Save failed')
})

els.composer.addEventListener('submit', event => {
  void startRun(event)
})

els.cancelRun.addEventListener('click', () => {
  void runUiAction(() => cancelRun(), 'Cancel failed')
})

els.recentProjects.addEventListener('click', event => {
  const target = event.target instanceof Element ? event.target : null
  const button = target?.closest('[data-root]')
  if (button) {
    void runUiAction(
      () => openProject(button.getAttribute('data-root')),
      'Open failed',
    )
  }
})

els.toolsList.addEventListener('click', event => {
  const target = event.target instanceof Element ? event.target : null
  const button = target?.closest('.tool-revoke')
  if (!button || !state.currentRoot) {
    return
  }

  void runUiAction(async () => {
    await api('/api/project/tools/revoke', {
      method: 'POST',
      body: JSON.stringify({
        root: state.currentRoot,
        toolId: button.getAttribute('data-tool-id'),
      }),
    })
    await refreshProject()
  }, 'Tool update failed')
})

window.addEventListener('beforeunload', () => {
  if (state.eventSource) {
    state.eventSource.close()
  }
})

async function bootstrap() {
  setProjectRootDisplay(projectsHomeRoot)
  await loadRecentProjects()
  const preferredProject =
    state.recentProjects.find(project =>
      project.root.startsWith(`${projectsHomeRoot}\\`),
    ) || state.recentProjects[0]

  if (preferredProject?.root) {
    await openProject(preferredProject.root)
  } else {
    state.currentRoot = ''
    setProjectRootDisplay(projectsHomeRoot)
    setIndicator('idle', 'Pick a project')
  }
}

void bootstrap()
