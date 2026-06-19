const { app, BrowserWindow, dialog, ipcMain } = require('electron')
const { execFileSync, spawn } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const PORT = process.env.BIOPLEASE_PORT || '4114'
const ROOT = path.resolve(__dirname, '..', '..', '..')
const SERVER_URL = `http://localhost:${PORT}`

let serverProcess = null

function resolveBunPath() {
  const candidates = [
    process.env.BUN_PATH,
    process.env.BUN_INSTALL
      ? path.join(process.env.BUN_INSTALL, 'bin', 'bun.exe')
      : null,
    process.env.USERPROFILE
      ? path.join(process.env.USERPROFILE, '.bun', 'bin', 'bun.exe')
      : null,
    process.env.LOCALAPPDATA
      ? path.join(process.env.LOCALAPPDATA, 'Programs', 'bun', 'bun.exe')
      : null,
  ].filter(Boolean)

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate
    }
  }

  try {
    const output = execFileSync('where.exe', ['bun'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    })
      .split(/\r?\n/)
      .map(line => line.trim())
      .find(line => line && fs.existsSync(line))
    if (output) {
      return output
    }
  } catch {}

  throw new Error(
    'Unable to find Bun. Set BUN_PATH or install Bun before launching the BioPLEASE desktop app.',
  )
}

async function waitForServer(url, timeoutMs = 20000) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok) {
        return
      }
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 300))
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function startServer() {
  if (serverProcess) {
    return
  }

  const bunPath = resolveBunPath()
  serverProcess = spawn(
    bunPath,
    ['run', 'src/bioplease/cli.ts', 'web', '--port', String(PORT)],
    {
      cwd: ROOT,
      env: {
        ...process.env,
        PORT: String(PORT),
      },
      stdio: 'inherit',
    },
  )

  serverProcess.on('exit', () => {
    serverProcess = null
  })
}

async function createWindow() {
  try {
    startServer()
  } catch (error) {
    await dialog.showErrorBox(
      'BioPLEASE Desktop Startup Failed',
      error instanceof Error ? error.message : String(error),
    )
    app.quit()
    return
  }
  await waitForServer(SERVER_URL)

  const window = new BrowserWindow({
    width: 1580,
    height: 980,
    minWidth: 1180,
    minHeight: 760,
    title: 'BioPLEASE',
    backgroundColor: '#090d14',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  window.removeMenu()
  await window.loadURL(SERVER_URL)
}

app.whenReady().then(createWindow)

ipcMain.handle('bioplease:pick-folder', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  })
  return result.canceled ? null : result.filePaths[0] || null
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (serverProcess) {
      serverProcess.kill()
      serverProcess = null
    }
    app.quit()
  }
})

app.on('before-quit', () => {
  if (serverProcess) {
    serverProcess.kill()
    serverProcess = null
  }
})
