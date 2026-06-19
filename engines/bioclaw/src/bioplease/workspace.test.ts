import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'

import { ensureProjectWorkspace } from './workspace.js'

const cleanup: string[] = []

afterEach(async () => {
  while (cleanup.length > 0) {
    const path = cleanup.pop()
    if (path) {
      await rm(path, { recursive: true, force: true })
    }
  }
})

describe('ensureProjectWorkspace', () => {
  test('creates the hidden .bioplease runtime contract and local skills/settings', async () => {
    const root = await mkdtemp(join(tmpdir(), 'bioplease-workspace-'))
    cleanup.push(root)

    const project = await ensureProjectWorkspace({
      projectRoot: root,
      title: 'Test Project',
      researchQuestion: 'How should this workspace be organized?',
      backgroundContext: 'Testing',
    })

    const config = JSON.parse(await readFile(project.paths.config, 'utf8'))
    expect(config.title).toBe('Test Project')
    expect(config.metadataDirectory).toBe('.bioplease')

    const claudeMd = await readFile(project.paths.projectClaude, 'utf8')
    expect(claudeMd).toContain('BioPLEASE Codex Contract')

    const plan = await readFile(project.paths.plan, 'utf8')
    expect(plan).toContain('# BioPLEASE Plan')

    const memory = await readFile(join(project.paths.memory, 'MEMORY.md'), 'utf8')
    expect(memory).toContain('# MEMORY')

    const settingsLocal = JSON.parse(
      await readFile(project.paths.settingsLocal, 'utf8'),
    )
    expect(settingsLocal.agentRouting.Execute).toBe('minimax-m2.7')
    expect(settingsLocal.agentRouting.Plan).toBe('gpt-5.4-mini')

    const mcpConfig = JSON.parse(await readFile(project.paths.mcpConfig, 'utf8'))
    expect(mcpConfig.mcpServers.biocontext_kb.url).toBeTruthy()

    const generatedSkill = await readFile(
      join(project.paths.skillsRoot, 'bioplease-learn', 'SKILL.md'),
      'utf8',
    )
    expect(generatedSkill).toContain('BioContextAI')
  })
})
