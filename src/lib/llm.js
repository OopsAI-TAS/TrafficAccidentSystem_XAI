import OpenAI from 'openai';
import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config(); // 환경 변수 로드

const DEFAULT_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

function getClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.');
  }
  return new OpenAI({ apiKey });
}

export async function generateDecision(messages) {
  const client = getClient();
  const response = await client.chat.completions.create({
    model: DEFAULT_MODEL,
    temperature: 0.7,
    messages: messages,
  });

  const text = response.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error('모델 응답이 비어있습니다.');
  return text;
}

export async function evaluateDecision(decision) {
  const geminiApiUrl = process.env.GEMINI_API_URL; // Set this in your .env file
  const geminiApiKey = process.env.GEMINI_API_KEY; // Set this in your .env file

  if (!geminiApiUrl || !geminiApiKey) {
    throw new Error('Gemini API URL or API Key is not configured.');
  }

  const response = await fetch(geminiApiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${geminiApiKey}`,
    },
    body: JSON.stringify({ input: decision }),
  });

  if (!response.ok) {
    throw new Error(`Gemini API error: ${response.statusText}`);
  }

  const result = await response.json();
  return result.evaluation; // Adjust based on Gemini's API response structure
}


