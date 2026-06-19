import { readdirSync } from 'fs'
import { join } from 'path'

import type { DoctorReport, RuntimeCheck } from './types.js'

function findRipgrepFallback(baseDir: string): string | null {
  try {
    const entries = readdirSync(baseDir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = join(baseDir, entry.name)
      if (entry.isDirectory()) {
        const nested = findRipgrepFallback(fullPath)
        if (nested) {
          return nested
        }
        continue
      }
      if (entry.isFile() && entry.name.toLowerCase() === 'rg.exe') {
        return fullPath
      }
    }
  } catch {
    return null
  }
  return null
}

export function resolveWorkingRipgrepPath(): string | null {
  const direct = Bun.which('rg')
  if (direct) {
    try {
      const result = Bun.spawnSync({
        cmd: [direct, '--version'],
        stderr: 'pipe',
        stdout: 'pipe',
        env: process.env,
      })
      if (result.exitCode === 0) {
        return direct
      }
    } catch {}
  }

  const localAppData = process.env.LOCALAPPDATA
  if (!localAppData) {
    return null
  }

  const wingetPackages = join(localAppData, 'Microsoft', 'WinGet', 'Packages')
  return findRipgrepFallback(wingetPackages)
}

function resolveCommandPath(command: string): string | null {
  if (command === 'bun' && process.execPath.toLowerCase().includes('bun')) {
    return process.execPath
  }
  if (command === 'rg') {
    return resolveWorkingRipgrepPath()
  }
  return Bun.which(command)
}

function runCheck(command: string, versionArgs: string[]): RuntimeCheck {
  const path = resolveCommandPath(command)

  if (!path) {
    return {
      name: command,
      available: false,
      command,
      path: null,
      detail: 'not found on PATH',
    }
  }

  let result:
    | ReturnType<typeof Bun.spawnSync>
    | null = null
  try {
    result = Bun.spawnSync({
      cmd: [path, ...versionArgs],
      stderr: 'pipe',
      stdout: 'pipe',
      env: process.env,
    })
  } catch (error) {
    return {
      name: command,
      available: false,
      command,
      path,
      detail:
        error instanceof Error ? error.message : `failed to execute ${command}`,
    }
  }

  const detail = `${new TextDecoder().decode(result.stdout).trim()} ${new TextDecoder().decode(result.stderr).trim()}`.trim()

  return {
    name: command,
    available: result.exitCode === 0,
    command,
    path,
    detail: detail || `exit=${result.exitCode}`,
  }
}

export function runDoctor(): DoctorReport {
  return {
    generatedAt: new Date().toISOString(),
    checks: [
      runCheck('bun', ['--version']),
      runCheck('node', ['--version']),
      runCheck('python', ['--version']),
      runCheck('docker', ['--version']),
      runCheck('rg', ['--version']),
    ],
  }
}

export function formatDoctorReport(report: DoctorReport): string {
  const lines = [`BioPLEASE doctor`, `Generated: ${report.generatedAt}`, '']
  for (const check of report.checks) {
    lines.push(
      `${check.available ? '[ok]' : '[missing]'} ${check.name} :: ${check.detail}`,
    )
  }
  return lines.join('\n')
}
