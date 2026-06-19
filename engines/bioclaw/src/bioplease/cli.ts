#!/usr/bin/env bun

import { Command } from 'commander'
import { spawn } from 'node:child_process'

import { formatDoctorReport, runDoctor } from './doctor.js'
import { getProjectSnapshot, openProjectFolder } from './projects.js'
import { runBioPleaseSession } from './runner.js'
import { startBioPleaseWebServer } from './web/server.js'
import { loadProject } from './workspace.js'

const program = new Command()

program
  .name('bioplease')
  .description('BioPLEASE Codex-style biomedical workspace built on top of OpenClaude')

program
  .command('doctor')
  .description('Check the local runtime required by the BioPLEASE/OpenClaude stack')
  .action(() => {
    const report = runDoctor()
    process.stdout.write(`${formatDoctorReport(report)}\n`)
  })

program
  .command('init')
  .description('Initialize BioPLEASE metadata inside a local working folder')
  .requiredOption('--project-root <path>', 'Local working folder')
  .option('--title <title>', 'Project title')
  .option('--question <question>', 'Research question')
  .option('--background <text>', 'Background context', '')
  .action(async options => {
    const project = await openProjectFolder({
      projectRoot: options.projectRoot,
      title: options.title,
      researchQuestion: options.question,
      backgroundContext: options.background,
    })
    process.stdout.write(`Initialized BioPLEASE project at ${project.root}\n`)
  })

program
  .command('status')
  .description('Print the current BioPLEASE snapshot for a local folder')
  .requiredOption('--project-root <path>', 'Local working folder')
  .action(async options => {
    const snapshot = await getProjectSnapshot(options.projectRoot)
    process.stdout.write(`${JSON.stringify(snapshot, null, 2)}\n`)
  })

program
  .command('web')
  .description('Start the BioPLEASE local web app')
  .option(
    '--port <port>',
    'Port for the BioPLEASE web app',
    value => Number.parseInt(value, 10),
    4114,
  )
  .action(async options => {
    startBioPleaseWebServer({ port: options.port })
    await new Promise(() => {})
  })

program
  .command('desktop')
  .description('Launch the BioPLEASE Electron desktop shell')
  .action(async () => {
    const electronBin = Bun.which('electron')
    if (!electronBin) {
      throw new Error('Electron is not installed yet. Run npm install first.')
    }
    const child = spawn(electronBin, ['src/bioplease/desktop/main.cjs'], {
      stdio: 'inherit',
    })
    await new Promise((resolve, reject) => {
      child.once('error', reject)
      child.once('close', resolve)
    })
  })

program
  .command('run')
  .description('Run the BioPLEASE phase workflow inside a local folder')
  .requiredOption('--project-root <path>', 'Local working folder')
  .requiredOption('--goal <goal>', 'Current execution goal')
  .option('--title <title>', 'Project title when bootstrapping a folder')
  .option('--question <question>', 'Research question when bootstrapping a folder')
  .option('--background <text>', 'Background context', '')
  .option(
    '--max-turns <turns>',
    'Maximum total turns across the phase workflow',
    value => Number.parseInt(value, 10),
    45,
  )
  .option('--permission-mode <mode>', 'Permission mode override', 'bypassPermissions')
  .action(async options => {
    const project = await loadProject(options.projectRoot).catch(() =>
      openProjectFolder({
        projectRoot: options.projectRoot,
        title: options.title,
        researchQuestion: options.question,
        backgroundContext: options.background,
      }),
    )

    const summary = await runBioPleaseSession({
      projectRoot: project.root,
      goal: options.goal,
      maxTurns: options.maxTurns,
      permissionMode: options.permissionMode,
      title: options.title,
      researchQuestion: options.question,
      backgroundContext: options.background,
      onEvent(event) {
        process.stdout.write(
          `[${event.timestamp}] ${event.phase ? `${event.phase} :: ` : ''}${event.type} :: ${event.message ?? ''}\n`,
        )
      },
    })

    process.stdout.write(
      [
        '',
        'BioPLEASE run complete',
        `Session ID: ${summary.sessionId}`,
        `Project: ${summary.projectRoot}`,
        `Transcript: ${summary.transcriptPath}`,
      ].join('\n') + '\n',
    )
  })

await program.parseAsync(process.argv)
