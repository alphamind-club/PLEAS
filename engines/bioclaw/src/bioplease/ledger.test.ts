import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'

import { writeWorkspaceLedger } from './ledger.js'
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

describe('writeWorkspaceLedger', () => {
  test('captures code and outputs from the opened folder rather than a nested workspace only', async () => {
    const root = await mkdtemp(join(tmpdir(), 'bioplease-ledger-'))
    cleanup.push(root)

    const project = await ensureProjectWorkspace({
      projectRoot: root,
      title: 'Ledger Project',
      researchQuestion: 'What did the agent create?',
    })

    await writeFile(join(root, 'pipeline.py'), 'print("ok")\n', 'utf8')
    await writeFile(join(project.paths.data, 'expression.csv'), 'gene,value\nFOXP3,1\n', 'utf8')
    await writeFile(join(project.paths.figures, 'plot.png'), 'fake', 'utf8')
    await writeFile(
      project.paths.plan,
      '# Plan\n\n- [ ] Finish the experiment\n- [x] Create the first pipeline\n',
      'utf8',
    )

    const ledger = await writeWorkspaceLedger(project)

    expect(ledger.primaryScripts).toContain('pipeline.py')
    expect(ledger.datasets).toContain('data/expression.csv')
    expect(ledger.figures).toContain('figures/plot.png')
    expect(ledger.unfinishedTasks).toContain('Finish the experiment')
    expect(ledger.currentWorkingDirectory).toBe(root)
  })
})
