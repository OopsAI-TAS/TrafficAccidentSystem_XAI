#!/usr/bin/env node
import 'dotenv/config';
import fs from 'fs';
import { readFile } from 'node:fs/promises';
import { stdin as input } from 'node:process';
import { buildMessages } from './lib/prompt.js';
import { validateInput } from './lib/validate.js';
import { generateDecision, evaluateDecision } from './lib/llm.js';

async function readInput(fileArg) {
  if (fileArg && fileArg !== '-') {
    return await readFile(fileArg, 'utf8');
  }
  const chunks = [];
  for await (const chunk of input) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function main() {
  const args = process.argv.slice(2);
  const flags = new Set(args.filter(a => a.startsWith('--')));
  const fileArg = args.find(a => !a.startsWith('--'));

  try {
    const raw = await readInput(fileArg);
    if (!raw || raw.trim().length === 0) {
      console.error('입력 JSON이 비어있습니다. 파일 경로를 인자로 주거나 표준입력으로 JSON을 전달하세요.');
      process.exit(1);
    }

    const parsed = JSON.parse(raw);
    const inputData = validateInput(parsed);
    const messages = buildMessages(inputData);

    if (flags.has('--debug')) {
      console.error(JSON.stringify({ messages }, null, 2));
    }

    const text = await generateDecision(messages);
    if (flags.has('--debug')) {
      console.error('Generated Decision:', text.trim());
    }

    try {
      const evaluation = await evaluateDecision(text.trim());
      process.stdout.write(evaluation.trim() + '\n');
    } catch (evalErr) {
      console.error('Evaluation 오류:', evalErr?.message || evalErr);
      process.exit(1);
    }
  } catch (err) {
    console.error('오류:', err?.message || err);
    process.exit(1);
  }
}

main();


