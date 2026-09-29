import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import os from 'os';
import { spawn } from 'child_process';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Determine port and host: prioritize CLI flags (--port/--host), then env vars (avoiding 8080 which is reserved for Nginx proxy)
const cliArgs = process.argv.slice(2);
let argPort: number | null = null;
let argHost: string | null = null;
for (let i = 0; i < cliArgs.length; i++) {
  if (cliArgs[i] === '--port' && cliArgs[i + 1]) {
    argPort = parseInt(cliArgs[i + 1], 10);
  }
  if (cliArgs[i] === '--host' && cliArgs[i + 1]) {
    argHost = cliArgs[i + 1];
  }
}

const PORT = argPort || (process.env.PORT && process.env.PORT !== '8080' ? parseInt(process.env.PORT, 10) : 3000);
const HOST = argHost || process.env.HOST || '0.0.0.0';

// Health check endpoint for container lifecycle
app.get('/health', (_req, res) => {
  res.status(200).json({ status: 'ok', uptime: process.uptime() });
});

const AGNES_BASE_URL = 'https://apihub.agnes-ai.com/v1';

// Server-side Gemini AI Client for high-fidelity speech synthesis and AI tasks
const geminiApiKey = process.env.GEMINI_API_KEY;
const ai = geminiApiKey
  ? new GoogleGenAI({
      apiKey: geminiApiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Helper to extract Agnes API key from request headers or environment
function getAgnesApiKey(req: express.Request): string | null {
  const customHeader = req.headers['x-agnes-key'] as string;
  if (customHeader && customHeader.trim()) {
    return customHeader.trim();
  }
  const authHeader = req.headers['authorization'];
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.substring(7).trim();
    if (token && token !== 'undefined' && token !== 'null') {
      return token;
    }
  }
  return process.env.AGNES_API_KEY || null;
}

// Resilient upstream fetch helper: prevents "Unexpected token '<'" by safely parsing JSON/HTML
interface UpstreamResult {
  ok: boolean;
  status: number;
  data?: any;
  error?: string;
  isRateLimit?: boolean;
  isQuota?: boolean;
}

async function safeUpstreamFetch(
  url: string,
  options: RequestInit,
  endpointName: string = 'Agnes AI'
): Promise<UpstreamResult> {
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch (netErr: any) {
    return {
      ok: false,
      status: 503,
      error: `Falha de conexão com os servidores da ${endpointName}: ${netErr.message || 'Servidor inalcançável'}`,
    };
  }

  const rawText = await response.text();
  let parsedJson: any = null;

  try {
    parsedJson = JSON.parse(rawText);
  } catch (_jsonErr) {
    // Non-JSON response (e.g. Cloudflare HTML 429/502/504 error page)
    const is429 =
      response.status === 429 ||
      rawText.includes('429') ||
      rawText.toLowerCase().includes('rate limit') ||
      rawText.toLowerCase().includes('too many requests');

    const isQuota =
      response.status === 402 ||
      rawText.toLowerCase().includes('insufficient') ||
      rawText.toLowerCase().includes('credits') ||
      rawText.toLowerCase().includes('quota');

    if (is429) {
      return {
        ok: false,
        status: 429,
        isRateLimit: true,
        error: `Limite de concorrência ou requisições atingido na ${endpointName} (HTTP 429). A plataforma de IA processa um vídeo por vez. Aguarde alguns segundos.`,
      };
    }

    if (isQuota) {
      return {
        ok: false,
        status: 402,
        isQuota: true,
        error: `Créditos ou saldo insuficientes na sua conta ${endpointName}. Verifique seu plano/saldo no painel da Agnes.`,
      };
    }

    const cleanSnippet = rawText
      .replace(/<[^>]*>?/gm, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 160);

    return {
      ok: false,
      status: response.status || 502,
      error: `A ${endpointName} retornou uma resposta não-JSON (HTTP ${response.status})${
        cleanSnippet ? `: ${cleanSnippet}` : ''
      }`,
    };
  }

  if (!response.ok) {
    const errorMsg =
      parsedJson?.error?.message ||
      parsedJson?.error ||
      parsedJson?.message ||
      `Erro HTTP ${response.status}`;
    const errorStr = String(errorMsg);
    const isRateLimit =
      response.status === 429 ||
      errorStr.toLowerCase().includes('rate limit') ||
      errorStr.toLowerCase().includes('too many requests') ||
      errorStr.toLowerCase().includes('concurrency') ||
      errorStr.toLowerCase().includes('concurren');
    const isQuota =
      response.status === 402 ||
      errorStr.toLowerCase().includes('credit') ||
      errorStr.toLowerCase().includes('balance') ||
      errorStr.toLowerCase().includes('quota') ||
      errorStr.toLowerCase().includes('insufficient');

    return {
      ok: false,
      status: response.status,
      isRateLimit,
      isQuota,
      data: parsedJson,
      error: isRateLimit
        ? `Limite de requisições excedido na ${endpointName} (429). Aguarde alguns segundos.`
        : isQuota
        ? `Saldo ou créditos insuficientes na sua conta ${endpointName}.`
        : errorMsg,
    };
  }

  return {
    ok: true,
    status: response.status,
    data: parsedJson,
  };
}

// 1. Text & Storyboard Scripting (agnes-2.5-flash)
app.post('/api/agnes/chat', async (req, res) => {
  try {
    const apiKey = getAgnesApiKey(req);
    if (!apiKey) {
      return res.status(401).json({
        error: 'Agnes API key não informada. Forneça nas configurações do app ou no .env',
      });
    }

    const { messages, model = 'agnes-2.5-flash', temperature = 0.7, max_tokens = 4096 } = req.body;

    const result = await safeUpstreamFetch(
      `${AGNES_BASE_URL}/chat/completions`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages,
          temperature,
          max_tokens,
        }),
      },
      'Agnes Chat'
    );

    if (!result.ok) {
      return res.status(result.status).json({
        error: result.error,
        isRateLimit: result.isRateLimit,
        isQuota: result.isQuota,
        data: result.data,
      });
    }

    return res.json(result.data);
  } catch (error: any) {
    console.error('Error in /api/agnes/chat:', error);
    return res.status(500).json({ error: error.message || 'Internal proxy error' });
  }
});

// Sanitizes and polishes Portuguese captions, instructions and voiceover texts
function cleanAndPolishPortugueseCaption(raw: string): string {
  if (!raw || typeof raw !== 'string') return '';
  let s = raw.trim();

  // 1. Fix "na para tigela começar" -> "na tigela para começar"
  s = s.replace(/\bna\s+para\s+(\w+)\s+(\w+)\b/gi, 'na $1 para $2');
  s = s.replace(/\bno\s+para\s+(\w+)\s+(\w+)\b/gi, 'no $1 para $2');
  s = s.replace(/\bna\s+para\b/gi, 'na');
  s = s.replace(/\bpara\s+na\b/gi, 'para a');
  s = s.replace(/\bno\s+para\b/gi, 'no');
  s = s.replace(/\bpara\s+no\b/gi, 'para o');
  s = s.replace(/\bde\s+para\b/gi, 'de');
  s = s.replace(/\bem\s+na\b/gi, 'na');
  s = s.replace(/\bem\s+no\b/gi, 'no');

  // 2. Fix misplaced conjunctions: "até e inflar dourar" -> "até inflar e dourar"
  s = s.replace(/\baté\s+e\s+(\w+)\s+(\w+)/gi, 'até $1 e $2');
  s = s.replace(/\baté\s+e\s+/gi, 'até ');
  s = s.replace(/\bobter\s+e\s+uma\b/gi, 'obter uma');
  s = s.replace(/\bobter\s+e\s+um\b/gi, 'obter um');
  s = s.replace(/\bficar\s+e\s+(\w+)/gi, 'ficar $1');

  // 3. Fix CTA connectors: "! e Salva me segue agora!" -> "! Salve e me siga agora!"
  s = s.replace(/[!.]\s*e\s+Salva\s+me\s+segue\s+agora!?/gi, '! Salve e me siga agora!');
  s = s.replace(/[!.]\s*e\s+salva\s+e\s+segue!?/gi, '! Salve e me siga!');
  s = s.replace(/\bSalva\s+me\s+segue\s+agora!?/gi, 'Salve e me siga agora!');
  s = s.replace(/\bSalva\s+me\s+segue\b/gi, 'Salve e me siga');
  s = s.replace(/\bcurte\s+e\s+salva\s+me\s+segue\b/gi, 'Curta, salve e me siga');

  // 4. Fix duplicated consecutive words (e.g., "queijo queijo", "a a", "o o")
  s = s.replace(/\b([a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]{2,})\s+\1\b/gi, '$1');

  // 5. Clean up multiple spaces and weird punctuation
  s = s.replace(/\s{2,}/g, ' ');
  s = s.replace(/\s+([,!?.])/g, '$1');

  // 6. Ensure first letter is capitalized
  if (s.length > 0) {
    s = s.charAt(0).toUpperCase() + s.slice(1);
  }

  return s;
}

// Dedicated Recipe Workflow Generation (Gemini 3.8 Flash structured output with Agnes AI fallback)
app.post('/api/recipe/generate', async (req, res) => {
  try {
    const {
      recipeNameOrIdea,
      stepCount = 5,
      aspectRatio = '9:16',
      handStyle,
      surfaceStyle,
      fixedAnchors,
    } = req.body;

    if (!recipeNameOrIdea) {
      return res.status(400).json({ error: 'Nome ou ideia da receita é obrigatório.' });
    }

    const handAnchor =
      fixedAnchors?.handAnchor ||
      handStyle ||
      'Mãos cuidadas do cozinheiro com unhas vermelhas e anel discreto';
    const countertopAnchor =
      fixedAnchors?.countertopAnchor ||
      surfaceStyle ||
      'Bancada de carvalho rústico escuro na cozinha com azulejos claros e clima acolhedor';
    const lightingAnchor =
      fixedAnchors?.lightingAnchor ||
      'Iluminação suave difusa de janela matinal 5600K com reflexos quentes e vapor aromático';
    const styleAnchor =
      fixedAnchors?.styleAnchor ||
      'Fotografia gastronômica comercial para Reels/TikTok, apetitosa em altíssima definição 8k';

    const targetSteps = typeof stepCount === 'number' && stepCount >= 3 ? stepCount : 8;

    // 1. Try server-side Gemini 3.8 Flash with structured JSON schema
    if (ai) {
      try {
        const geminiPrompt = `Você é um diretor de produção audiovisual e criador de conteúdo gastronômico viral para Reels, TikTok e YouTube Shorts.
Crie um roteiro culinário contínuo, ágil e altamente profissional com EXATAMENTE ${targetSteps} PASSOS/TOMADAS SEQUENCIAIS (sem pular ou agrupar etapas) utilizando a técnica de Last-Frame Continuity.
Receita alvo: "${recipeNameOrIdea}"
Formato de tela: ${aspectRatio}

ÂNCORAS FIXAS (Devem estar sempre presentes no visual de cada tomada):
- Mãos: "${handAnchor}"
- Bancada: "${countertopAnchor}"
- Iluminação: "${lightingAnchor}"
- Estilo: "${styleAnchor}"

REGRA DE CONSISTÊNCIA VISUAL DE INGREDIENTES E CORES (EXTREMAMENTE CRÍTICA):
- Identifique os ingredientes-chave da receita (ex: tipo de molho, formato da massa, cor da proteína).
- MANTENHA A MESMA COR E TIPO DE INGREDIENTES DO INÍCIO AO FIM!
- Se a receita usa molho de tomate vermelho Pomodoro, TODOS os prompts seguintes DEVEM especificar "rich red vibrant tomato sauce". NUNCA altere para molho branco ou quatro queijos no meio da receita!
- Se a massa for penne ou espaguete, mantenha rigorosamente a mesma massa até a garfada final.

REGRA DE LOCUÇÃO RÁPIDA E DIRETA (ESTILO TIKTOK / REELS 1.25x):
- O campo "voiceoverText" de CADA passo DEVE ser curto, direto, enérgico e imperativo (MÁXIMO 8 A 12 PALAVRAS POR PASSO).
- Proibido pausas longas, reticências ("...") ou introduções lentas como "Sabe... a vovó testava...".
- Exemplos de frases perfeitas:
  - "Ferva a água com bastante sal e coloque a massa!"
  - "Pique o alho fininho na nossa tábua de madeira."
  - "Azeite bem quente e doure até perfumar a cozinha!"
  - "Despeje os camarões suculentos e salteie no fogo vivo."
  - "Emprate com pinça para aquele toque de restaurante."
  - "Olha que cremosidade surreal! Salve agora e me siga!"

REGRA DE GRAMÁTICA PERFEITA E NATURALIDADE (PORTUGUÊS DO BRASIL):
- A sintaxe deve ser 100% impecável, sem repetições de palavras ou preposições encavaladas.
- ESTRITAMENTE PROIBIDO:
  * Repetições de preposições (ex: NUNCA gere "na para tigela", gere "na tigela para começar" ou "para a tigela").
  * Conjunções deslocadas ou soltas (ex: NUNCA gere "até e inflar dourar", gere "até inflar e dourar intensamente"; NUNCA "obter e uma massa", gere "obter uma massa lisa").
  * Duplicações de palavras consecutivas (ex: "queijo queijo", "de de").
  * Conectores estranhos em CTA (ex: use "Salve agora e me siga!" ou "Salve para não perder!").
- Garanta que as legendas e a locução fiquem 100% profissionais, elegantes e naturais para Reels/TikTok.

ESTRUTURA OBRIGATÓRIA DE ${targetSteps} TOMADAS ÁGEIS:
- Se ${targetSteps} >= 7 ou 8 tomadas, decupe detalhadamente o passo a passo com micro-ações apetitosas:
  1. Ingrediente principal na tigela ou corte inicial na tábua de madeira (prep_cutting / cutting_board)
  2. Adição de temperos, ervas frescas ou aromáticos (preparo / bowl_prep)
  3. Adição de líquidos, queijos ou farinhas e incorporação suave (bowl_prep)
  4. Misturar/sovar ou levar ao fogo vivo/frigideira/air fryer (saute_frying ou appliance_enter)
  5. Ponto de cozimento, borbulhamento ou douramento crocante com fumaça subindo (pan_stove ou appliance_exit)
  6. Finalização e montagem no prato de cerâmica ou travessa (plated_hero / serving_plate)
  7. Pegar com garfo ou pinça / Partir ao meio com queijo esticando (tasting / tasting_fork)
  8. Tomada final de encerramento com CTA viral para curtir, salvar e seguir (social_cta / social_hero)

MUITA ATENÇÃO AOS PROMPTS DE IMAGEM:
- O array "steps" no JSON retornado DEVE CONTER EXATAMENTE ${targetSteps} OBJETOS!
- imagePrompt (Frame Inicial / 0s) e endImagePrompt (Frame Final / parada suave) devem reforçar os mesmos ingredientes, mesma louça e mesma iluminação.
- Retorne estritamente o JSON preenchido conforme o esquema.`;

        const geminiResponse = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: geminiPrompt,
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                category: { type: Type.STRING },
                prepTime: { type: Type.STRING },
                aspectRatio: { type: Type.STRING },
                fixedAnchors: {
                  type: Type.OBJECT,
                  properties: {
                    handAnchor: { type: Type.STRING },
                    countertopAnchor: { type: Type.STRING },
                    lightingAnchor: { type: Type.STRING },
                    styleAnchor: { type: Type.STRING },
                  },
                  required: ['handAnchor', 'countertopAnchor', 'lightingAnchor', 'styleAnchor'],
                },
                handStyle: { type: Type.STRING },
                bowlStyle: { type: Type.STRING },
                surfaceStyle: { type: Type.STRING },
                steps: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      stepNumber: { type: Type.INTEGER },
                      stepType: { type: Type.STRING },
                      equipmentStation: { type: Type.STRING },
                      utensil: { type: Type.STRING },
                      continuityRule: { type: Type.STRING },
                      continuityNote: { type: Type.STRING },
                      actionTitle: { type: Type.STRING },
                      instruction: { type: Type.STRING },
                      voiceoverText: { type: Type.STRING },
                      sfx: { type: Type.STRING },
                      imagePrompt: { type: Type.STRING },
                      endImagePrompt: { type: Type.STRING },
                      videoPrompt: { type: Type.STRING },
                      durationSeconds: { type: Type.INTEGER },
                    },
                    required: [
                      'stepNumber',
                      'stepType',
                      'equipmentStation',
                      'utensil',
                      'actionTitle',
                      'instruction',
                      'voiceoverText',
                      'sfx',
                      'imagePrompt',
                      'videoPrompt',
                      'durationSeconds',
                    ],
                  },
                },
              },
              required: ['title', 'category', 'prepTime', 'aspectRatio', 'fixedAnchors', 'steps'],
            },
          },
        });

        const jsonText = geminiResponse.text?.trim();
        if (jsonText) {
          const parsed = JSON.parse(jsonText);
          if (Array.isArray(parsed.steps)) {
            parsed.steps = parsed.steps.map((st: any) => ({
              ...st,
              instruction: cleanAndPolishPortugueseCaption(st.instruction),
              voiceoverText: cleanAndPolishPortugueseCaption(st.voiceoverText),
              actionTitle: cleanAndPolishPortugueseCaption(st.actionTitle),
            }));
          }
          return res.json({ ok: true, source: 'gemini', data: parsed });
        }
      } catch (geminiErr: any) {
        console.warn('Gemini structured recipe generation fallback:', geminiErr?.message || geminiErr);
      }
    }

    // 2. Fallback to Agnes Chat if key provided
    const apiKey = getAgnesApiKey(req);
    if (!apiKey) {
      return res.status(401).json({
        error: 'Chave de API não informada para geração de receita.',
      });
    }

    const agnesResult = await safeUpstreamFetch(
      `${AGNES_BASE_URL}/chat/completions`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'agnes-2.5-flash',
          messages: [
            {
              role: 'system',
              content:
                'Você é um diretor especializado em vídeos virais de culinária com IA. Responda exclusivamente em formato JSON bem formado, fechando todas as chaves e colchetes. Cada passo DEVE ser um objeto começando com { e terminando com }.',
            },
            {
              role: 'user',
              content: `Roteiro para receita: "${recipeNameOrIdea}". ${stepCount} passos, formato ${aspectRatio}. Mãos: "${handAnchor}", Bancada: "${countertopAnchor}", Luz: "${lightingAnchor}". Responda em JSON com title, category, prepTime, aspectRatio, fixedAnchors e array steps (com stepNumber, stepType, equipmentStation, utensil, actionTitle, instruction, voiceoverText, sfx, imagePrompt, videoPrompt, durationSeconds).`,
            },
          ],
          temperature: 0.7,
          max_tokens: 4096,
        }),
      },
      'Agnes Recipe'
    );

    if (!agnesResult.ok) {
      return res.status(agnesResult.status).json({
        error: agnesResult.error,
        isRateLimit: agnesResult.isRateLimit,
        isQuota: agnesResult.isQuota,
      });
    }

    const rawContent = agnesResult.data?.choices?.[0]?.message?.content || '';
    return res.json({ ok: true, source: 'agnes', raw: rawContent });
  } catch (error: any) {
    console.error('Error in /api/recipe/generate:', error);
    return res.status(500).json({ error: error.message || 'Internal proxy error' });
  }
});

// 2. Image Generation & Editing (agnes-image-2.0-flash)
app.post('/api/agnes/images/generations', async (req, res) => {
  try {
    const apiKey = getAgnesApiKey(req);
    if (!apiKey) {
      return res.status(401).json({
        error: 'Agnes API key não informada. Forneça nas configurações do app ou no .env',
      });
    }

    const {
      model = 'agnes-image-2.0-flash',
      prompt,
      size = '1024x768',
      return_base64 = false,
      extra_body,
    } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt é obrigatório.' });
    }

    const requestPayload: any = {
      model,
      prompt,
      size,
    };

    if (return_base64) {
      requestPayload.return_base64 = true;
    }

    if (extra_body) {
      requestPayload.extra_body = extra_body;
    } else {
      requestPayload.extra_body = { response_format: 'url' };
    }

    const result = await safeUpstreamFetch(
      `${AGNES_BASE_URL}/images/generations`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      },
      'Agnes Image'
    );

    if (!result.ok) {
      return res.status(result.status).json({
        error: result.error,
        isRateLimit: result.isRateLimit,
        isQuota: result.isQuota,
        data: result.data,
      });
    }

    return res.json(result.data);
  } catch (error: any) {
    console.error('Error in /api/agnes/images/generations:', error);
    return res.status(500).json({ error: error.message || 'Internal proxy error' });
  }
});

// 3. Create Video Task (agnes-video-v2.0)
app.post('/api/agnes/videos', async (req, res) => {
  try {
    const apiKey = getAgnesApiKey(req);
    if (!apiKey) {
      return res.status(401).json({
        error: 'Agnes API key não informada. Forneça nas configurações do app ou no .env',
      });
    }

    const {
      model = 'agnes-video-v2.0',
      prompt,
      image,
      first_frame_image,
      last_frame_image,
      secondKeyframeUrl,
      mode,
      height = 768,
      width = 1152,
      num_frames = 121,
      frame_rate = 24,
      seed,
      negative_prompt,
      extra_body,
    } = req.body;

    const requestPayload: any = {
      model,
      prompt: prompt || 'Cinematic smooth camera motion, high fidelity, 4k detail',
      height,
      width,
      num_frames,
      frame_rate,
    };

    // Dual keyframe interpolation support (First-Frame + End-Frame)
    const startImg = first_frame_image || (Array.isArray(image) ? image[0] : image);
    const endImg = last_frame_image || secondKeyframeUrl || (Array.isArray(image) && image.length > 1 ? image[1] : undefined);

    if (startImg && endImg) {
      requestPayload.image = [startImg, endImg];
      requestPayload.mode = 'keyframes';
      requestPayload.extra_body = {
        ...(extra_body || {}),
        image: [startImg, endImg],
        mode: 'keyframes',
        first_frame_image: startImg,
        last_frame_image: endImg,
      };
    } else if (image) {
      requestPayload.image = image;
    } else if (startImg) {
      requestPayload.image = startImg;
    }

    if (mode && !requestPayload.mode) {
      requestPayload.mode = mode;
    }
    if (seed !== undefined && seed !== null) {
      requestPayload.seed = seed;
    }
    if (negative_prompt) {
      requestPayload.negative_prompt = negative_prompt;
    }
    if (extra_body) {
      requestPayload.extra_body = extra_body;
    }

    const result = await safeUpstreamFetch(
      `${AGNES_BASE_URL}/videos`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      },
      'Agnes Video'
    );

    if (!result.ok) {
      return res.status(result.status).json({
        error: result.error,
        isRateLimit: result.isRateLimit,
        isQuota: result.isQuota,
        data: result.data,
      });
    }

    return res.json(result.data);
  } catch (error: any) {
    console.error('Error in /api/agnes/videos:', error);
    return res.status(500).json({ error: error.message || 'Internal proxy error' });
  }
});

// 4. Query Video Status & Result
app.get('/api/agnes/videos/status', async (req, res) => {
  try {
    const apiKey = getAgnesApiKey(req);
    if (!apiKey) {
      return res.status(401).json({
        error: 'Agnes API key não informada.',
      });
    }

    const videoId = req.query.video_id as string;
    const taskId = req.query.task_id as string;
    const modelName = (req.query.model_name as string) || 'agnes-video-v2.0';

    if (!videoId && !taskId) {
      return res.status(400).json({ error: 'video_id ou task_id é obrigatório.' });
    }

    const primaryId = taskId || videoId;

    // Try standard endpoint /v1/videos/<TASK_ID>
    let result = await safeUpstreamFetch(
      `${AGNES_BASE_URL}/videos/${encodeURIComponent(primaryId)}`,
      {
        method: 'GET',
        headers: { Authorization: `Bearer ${apiKey}` },
      },
      'Agnes Video Status'
    );

    // If 404 or failed on /v1/videos, fallback to /agnesapi
    if (!result.ok) {
      const altUrl = `https://apihub.agnes-ai.com/agnesapi?video_id=${encodeURIComponent(
        videoId || taskId
      )}&model_name=${encodeURIComponent(modelName)}`;
      const altResult = await safeUpstreamFetch(
        altUrl,
        {
          method: 'GET',
          headers: { Authorization: `Bearer ${apiKey}` },
        },
        'Agnes Video Status'
      );
      if (altResult.ok) {
        result = altResult;
      }
    }

    if (!result.ok) {
      return res.status(result.status).json({
        error: result.error,
        isRateLimit: result.isRateLimit,
        isQuota: result.isQuota,
      });
    }

    const raw = result.data || {};
    // Normalize status, progress and videoUrl
    let status = raw.status;
    if (!status && raw.internal_status) {
      status = raw.internal_status === 'inference' ? 'in_progress' : raw.internal_status;
    }
    if (raw.completed_at && !status) {
      status = 'completed';
    }

    const progress =
      typeof raw.progress === 'number'
        ? raw.progress
        : typeof raw.internal_progress === 'number'
        ? raw.internal_progress
        : status === 'completed'
        ? 100
        : 0;

    const videoUrl = raw.url || raw.video_url || raw.metadata?.url || null;

    return res.json({
      ...raw,
      status: status || 'in_progress',
      progress,
      video_url: videoUrl,
      url: videoUrl,
    });
  } catch (error: any) {
    console.error('Error in /api/agnes/videos/status:', error);
    return res.status(500).json({ error: error.message || 'Internal proxy error' });
  }
});

// 5. Proxy media to prevent browser canvas CORS taint when stitching/exporting and support video seek
app.get('/api/proxy-media', async (req, res) => {
  try {
    const mediaUrl = req.query.url as string;
    if (!mediaUrl || !mediaUrl.startsWith('http')) {
      return res.status(400).json({ error: 'URL de mídia inválida' });
    }

    const rangeHeader = req.headers.range;
    const fetchHeaders: Record<string, string> = {};
    if (rangeHeader) {
      fetchHeaders['Range'] = rangeHeader;
    }

    const response = await fetch(mediaUrl, { headers: fetchHeaders });
    if (!response.ok && response.status !== 206) {
      return res.status(response.status).send('Falha ao baixar mídia');
    }

    const contentType = response.headers.get('content-type') || 'video/mp4';
    const contentRange = response.headers.get('content-range');
    const contentLength = response.headers.get('content-length');
    const acceptRanges = response.headers.get('accept-ranges') || 'bytes';

    res.setHeader('Content-Type', contentType);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Range, Content-Type');
    res.setHeader('Accept-Ranges', acceptRanges);
    res.setHeader('Cache-Control', 'public, max-age=86400');

    if (contentRange) res.setHeader('Content-Range', contentRange);
    if (contentLength) res.setHeader('Content-Length', contentLength);

    res.status(response.status);

    const arrayBuffer = await response.arrayBuffer();
    return res.send(Buffer.from(arrayBuffer));
  } catch (err: any) {
    console.error('Error in /api/proxy-media:', err);
    return res.status(500).json({ error: err.message || 'Media proxy error' });
  }
});

// 6. Test Key endpoint
app.post('/api/agnes/test-key', async (req, res) => {
  try {
    const apiKey = getAgnesApiKey(req);
    if (!apiKey) {
      return res.status(400).json({ valid: false, message: 'Nenhuma chave fornecida.' });
    }

    // Send a minimal test chat completion
    const testResult = await safeUpstreamFetch(
      `${AGNES_BASE_URL}/chat/completions`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'agnes-2.5-flash',
          messages: [{ role: 'user', content: 'ping' }],
          max_tokens: 5,
        }),
      },
      'Agnes Key Test'
    );

    if (testResult.ok) {
      return res.json({ valid: true, message: 'Chave Agnes AI validada com sucesso!' });
    } else {
      return res.status(testResult.status).json({
        valid: false,
        message: testResult.error || `Erro HTTP ${testResult.status}`,
      });
    }
  } catch (error: any) {
    return res.status(500).json({ valid: false, message: error.message });
  }
});

// In-memory cache for synthesized voiceover speech to protect Gemini free tier quota
const ttsCache = new Map<string, { audio: string; mimeType: string; sampleRate: number }>();

// ==========================================
// UNIFIED RECIPE BACKGROUND JOB MANAGER (READY FOR ANDROID KOTLIN & WEB)
// ==========================================
export interface RecipeJobStep {
  stepNumber: number;
  station: string;
  actionTitle: string;
  instruction: string;
  voiceoverText: string;
  sfx: string;
  imagePrompt: string;
  videoPrompt: string;
  imageUrl?: string;
  videoId?: string;
  videoUrl?: string;
  status: 'pending' | 'generating_image' | 'generating_video' | 'completed' | 'failed';
  error?: string;
}

export interface RecipeJob {
  id: string;
  createdAt: string;
  updatedAt: string;
  dishName: string;
  category: string;
  aspectRatio: '9:16' | '16:9' | '1:1';
  fixedAnchors: {
    hands: string;
    countertop: string;
    lighting: string;
  };
  status: 'queued' | 'scripting' | 'processing' | 'completed' | 'failed';
  progress: number; // 0 to 100
  currentStepIndex: number;
  totalSteps: number;
  message: string;
  steps: RecipeJobStep[];
  stitchedVideoUrl?: string;
  error?: string;
}

const recipeJobs = new Map<string, RecipeJob>();

// Clean up stale jobs older than 6 hours
setInterval(() => {
  const sixHoursAgo = Date.now() - 6 * 60 * 60 * 1000;
  for (const [id, job] of recipeJobs.entries()) {
    if (new Date(job.createdAt).getTime() < sixHoursAgo) {
      recipeJobs.delete(id);
    }
  }
}, 30 * 60 * 1000);

/**
 * Background worker that steps through recipe video generation
 */
async function runRecipeJobPipeline(jobId: string, apiKey: string) {
  const job = recipeJobs.get(jobId);
  if (!job) return;

  try {
    // STEP 1: Generate Script with Gemini or fallback
    job.status = 'scripting';
    job.message = 'Gerando roteiro técnico e ângulos cinematográficos com IA...';
    job.progress = 10;
    job.updatedAt = new Date().toISOString();

    const handAnchor = job.fixedAnchors.hands;
    const countertopAnchor = job.fixedAnchors.countertop;
    const lightingAnchor = job.fixedAnchors.lighting;

    let parsedRecipe: any = null;

    if (ai) {
      try {
        const geminiResponse = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: [
            {
              role: 'user',
              parts: [
                {
                  text: `Você é um diretor de produção audiovisual gastronômica. Crie um roteiro de 5 passos para a receita: "${job.dishName}".
Âncoras Fixas Obrigatórias:
- Mãos: "${handAnchor}"
- Bancada: "${countertopAnchor}"
- Iluminação: "${lightingAnchor}"

Responda em formato estruturado com 5 passos lógicos:
Passo 1: Tábua de corte (preparo)
Passo 2: Frigideira no fogão (refogado)
Passo 3: Adição do ingrediente principal
Passo 4: Empratamento em cerâmica artesanal
Passo 5: Garfada macro irresistível + CTA social`,
                },
              ],
            },
          ],
          config: {
            temperature: 0.7,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                category: { type: Type.STRING },
                prepTime: { type: Type.STRING },
                steps: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      stepNumber: { type: Type.INTEGER },
                      station: { type: Type.STRING },
                      actionTitle: { type: Type.STRING },
                      instruction: { type: Type.STRING },
                      voiceoverText: { type: Type.STRING },
                      sfx: { type: Type.STRING },
                      imagePrompt: { type: Type.STRING },
                      videoPrompt: { type: Type.STRING },
                    },
                    required: [
                      'stepNumber',
                      'station',
                      'actionTitle',
                      'instruction',
                      'voiceoverText',
                      'sfx',
                      'imagePrompt',
                      'videoPrompt',
                    ],
                  },
                },
              },
              required: ['title', 'category', 'prepTime', 'steps'],
            },
          },
        });

        const scriptText = geminiResponse.text?.trim();
        if (scriptText) {
          parsedRecipe = JSON.parse(scriptText);
        }
      } catch (err: any) {
        console.warn(`[Job ${jobId}] Gemini script generation fallback:`, err?.message || err);
      }
    }

    // Default template fallback if AI script generation failed
    if (!parsedRecipe || !Array.isArray(parsedRecipe.steps) || parsedRecipe.steps.length === 0) {
      parsedRecipe = {
        title: job.dishName,
        category: 'Gastronomia',
        steps: [
          {
            stepNumber: 1,
            station: 'cutting_board',
            actionTitle: 'Corte e Mise en Place',
            instruction: 'Fatiar os ingredientes frescos na tábua de madeira.',
            voiceoverText: `Para começar nosso ${job.dishName}, fatiamos tudo bem fininho com muito capricho!`,
            sfx: 'chop',
            imagePrompt: `Top-down macro shot, chef cutting fresh ingredients for ${job.dishName} on dark oak cutting board. ${handAnchor}, ${lightingAnchor}, commercial food 8k.`,
            videoPrompt: `Smooth downward camera tracking, sharp chef knife finely chopping aromatics on wooden board. Realistic slice motion.`,
          },
          {
            stepNumber: 2,
            station: 'pan_stove',
            actionTitle: 'Refogado Aromático',
            instruction: 'Dourar os temperos no azeite quente na frigideira sobre chama acesa.',
            voiceoverText: 'Em seguida, frigideira quente no fogo com azeite e aquele chiado delicioso!',
            sfx: 'sizzle',
            imagePrompt: `Stainless steel skillet on active gas stove burner with blue flames. Fresh ingredients sizzling in olive oil. ${countertopAnchor}, ${lightingAnchor}, 8k.`,
            videoPrompt: `Cinematic close-up tracking shot into the sizzling hot pan. Gentle stirring with kitchen tongs, steam rising realistically.`,
          },
          {
            stepNumber: 3,
            station: 'pan_stove',
            actionTitle: 'Ingrediente Principal',
            instruction: 'Adicionar a proteína suculenta ao refogado borbulhante.',
            voiceoverText: 'Agora adicionamos o protagonista da receita, dourando por igual!',
            sfx: 'sizzle',
            imagePrompt: `Close-up shot of main ingredients being added to the hot bubbling skillet. ${handAnchor}, golden reflections, natural steam, 8k food photography.`,
            videoPrompt: `Dynamic slow motion 24fps as ingredients sear with aromatic oil drops and steam, rich caramelized textures.`,
          },
          {
            stepNumber: 4,
            station: 'serving_plate',
            actionTitle: 'Empratamento de Restaurante',
            instruction: 'Transferir cuidadosamente o prato finalizado para a cerâmica artesanal.',
            voiceoverText: 'Hora de empratar: montamos com pinça para aquele visual de alta gastronomia.',
            sfx: 'stir',
            imagePrompt: `Chef hands with kitchen tweezers plating ${job.dishName} into a wide shallow artisan ceramic bowl. ${countertopAnchor}, ${lightingAnchor}, 8k.`,
            videoPrompt: `Elegant macro tilt down, tweezers placing the final garnish. Glossy sauce finish, gentle swirl motion.`,
          },
          {
            stepNumber: 5,
            station: 'tasting_fork',
            actionTitle: 'Garfada Irresistível & CTA',
            instruction: 'Close macro da garfada com fumaça e convite para curtir e seguir.',
            voiceoverText: 'Olha que cremosidade e textura surreal! Já salva essa receita e me segue para mais!',
            sfx: 'timer_bell',
            imagePrompt: `Extreme macro close-up fork lifting a perfect steaming bite of ${job.dishName}. Shallow depth of field, appetizing texture, 8k commercial.`,
            videoPrompt: `Slow cinematic lift of the fork, savory juices and warm steam curling up towards the camera. Mouthwatering hero shot.`,
          },
        ],
      };
    }

    job.steps = parsedRecipe.steps.map((st: any, i: number) => ({
      stepNumber: st.stepNumber || i + 1,
      station: st.station || 'cooking_station',
      actionTitle: st.actionTitle || `Etapa ${i + 1}`,
      instruction: st.instruction || '',
      voiceoverText: st.voiceoverText || '',
      sfx: st.sfx || 'sizzle',
      imagePrompt: st.imagePrompt || '',
      videoPrompt: st.videoPrompt || '',
      status: 'pending',
    }));

    job.totalSteps = job.steps.length;
    job.status = 'processing';
    job.progress = 20;

    // STEP 2: Process Steps Sequentially (Real or Demo fallback)
    const hasValidKey = Boolean(apiKey && apiKey.trim().length > 8);

    for (let i = 0; i < job.steps.length; i++) {
      const step = job.steps[i];
      job.currentStepIndex = i;
      job.message = `Processando Passo ${i + 1}/${job.totalSteps}: ${step.actionTitle}...`;
      job.updatedAt = new Date().toISOString();

      // Step Image Generation
      step.status = 'generating_image';
      job.progress = 20 + Math.round((i / job.totalSteps) * 70);

      if (hasValidKey) {
        try {
          // Generate image with Agnes API
          const imgRes = await safeUpstreamFetch(
            `${AGNES_BASE_URL}/images/generations`,
            {
              method: 'POST',
              headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model: 'agnes-image-2.0-flash',
                prompt: step.imagePrompt,
                size: job.aspectRatio === '9:16' ? '768x1024' : '1024x768',
                extra_body: { response_format: 'url' },
              }),
            },
            'Agnes Image'
          );

          if (imgRes.ok && imgRes.data?.data?.[0]?.url) {
            step.imageUrl = imgRes.data.data[0].url;
          }
        } catch (imgErr) {
          console.warn(`[Job ${jobId}] Image gen upstream issue:`, imgErr);
        }
      }

      // Fallback local sample asset if no key or error
      if (!step.imageUrl) {
        const demoAssets = [
          '/src/assets/images/pasta_step1_chopping_1790389305979.jpg',
          '/src/assets/images/pasta_step2_stove_1790389315877.jpg',
          '/src/assets/images/pasta_step3_shrimp_1790389326020.jpg',
          '/src/assets/images/pasta_step4_plating_1790389337204.jpg',
          '/src/assets/images/pasta_step5_tasting_1790389347531.jpg',
        ];
        step.imageUrl = demoAssets[i % demoAssets.length];
      }

      // Step Video Generation with Dual Keyframes (Start ➔ End)
      step.status = 'generating_video';
      job.message = `Animando tomada ${i + 1}/${job.totalSteps} com interpolação de quadros-chave duplos...`;
      job.updatedAt = new Date().toISOString();

      if (hasValidKey && step.imageUrl?.startsWith('http')) {
        try {
          const videoPayload: any = {
            model: 'agnes-video-v2.0',
            prompt: step.videoPrompt || 'Cinematic culinary motion flowing smoothly to final resting pose',
            image_url: step.imageUrl,
            aspect_ratio: job.aspectRatio,
            duration: 3,
          };

          const videoRes = await safeUpstreamFetch(
            `${AGNES_BASE_URL}/videos`,
            {
              method: 'POST',
              headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(videoPayload),
            },
            'Agnes Video'
          );

          if (videoRes.ok) {
            const taskId = videoRes.data?.task_id || videoRes.data?.video_id || videoRes.data?.id;
            step.videoId = taskId;

            // Poll for completion (up to 30 attempts, 5s delay)
            let attempts = 0;
            while (attempts < 30) {
              await new Promise((r) => setTimeout(r, 5000));
              attempts++;

              const pollRes = await safeUpstreamFetch(
                `${AGNES_BASE_URL}/videos/${encodeURIComponent(taskId)}`,
                {
                  method: 'GET',
                  headers: { Authorization: `Bearer ${apiKey}` },
                },
                'Agnes Video Poll'
              );

              if (pollRes.ok) {
                const status = pollRes.data?.status || pollRes.data?.internal_status;
                const vUrl = pollRes.data?.url || pollRes.data?.video_url;
                if (status === 'completed' || vUrl) {
                  step.videoUrl = vUrl;
                  break;
                }
                if (status === 'failed') {
                  break;
                }
              }
            }
          }
        } catch (vidErr) {
          console.warn(`[Job ${jobId}] Video gen error:`, vidErr);
        }
      }

      step.status = 'completed';
    }

    job.status = 'completed';
    job.progress = 100;
    job.message = 'Vídeo de receita gerado com sucesso!';
    // Expose primary video or fallback
    job.stitchedVideoUrl =
      job.steps.find((s) => s.videoUrl)?.videoUrl ||
      job.steps[job.steps.length - 1]?.imageUrl ||
      '/src/assets/images/pasta_step5_tasting_1790389347531.jpg';
    job.updatedAt = new Date().toISOString();
  } catch (error: any) {
    console.error(`[Job ${jobId}] Pipeline failed:`, error);
    job.status = 'failed';
    job.error = error.message || 'Erro durante a orquestração do pipeline.';
    job.message = 'Falha no processamento da receita.';
    job.updatedAt = new Date().toISOString();
  }
}

// REST ENDPOINT: 1. Start Recipe Video Generation Pipeline (Called by Android Kotlin & Web)
app.post('/api/recipes/pipeline/start', async (req, res) => {
  try {
    const {
      dishName,
      category = 'Gastronomia',
      aspectRatio = '9:16',
      fixedAnchors = {},
    } = req.body;

    if (!dishName || typeof dishName !== 'string' || !dishName.trim()) {
      return res.status(400).json({ error: 'O nome do prato (dishName) é obrigatório.' });
    }

    const apiKey = getAgnesApiKey(req) || '';
    const jobId = `rec_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;

    const newJob: RecipeJob = {
      id: jobId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      dishName: dishName.trim(),
      category: category || 'Gastronomia',
      aspectRatio: aspectRatio === '16:9' || aspectRatio === '1:1' ? aspectRatio : '9:16',
      fixedAnchors: {
        hands: fixedAnchors.hands || 'Mãos femininas cuidadas, unhas vermelhas e anel discreto',
        countertop: fixedAnchors.countertop || 'Bancada rústica de carvalho escuro',
        lighting: fixedAnchors.lighting || 'Luz difusa de janela matinal 5600K suave',
      },
      status: 'queued',
      progress: 0,
      currentStepIndex: 0,
      totalSteps: 5,
      message: 'Receita enfileirada para processamento...',
      steps: [],
    };

    recipeJobs.set(jobId, newJob);

    // Fire & forget background execution so client (Android or Web) gets instant 202 Accepted response
    runRecipeJobPipeline(jobId, apiKey);

    return res.status(202).json({
      ok: true,
      jobId,
      status: 'queued',
      message: 'Tarefa de geração iniciada em segundo plano.',
      statusEndpoint: `/api/recipes/pipeline/status/${jobId}`,
    });
  } catch (err: any) {
    console.error('Error starting recipe pipeline:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
});

// REST ENDPOINT: 2. Query Status of a Recipe Job (Polling endpoint for Android Retrofit & Web)
app.get('/api/recipes/pipeline/status/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = recipeJobs.get(jobId);

  if (!job) {
    return res.status(404).json({ error: 'Trabalho de receita não encontrado ou expirado.' });
  }

  return res.json({
    jobId: job.id,
    status: job.status,
    progress: job.progress,
    message: job.message,
    dishName: job.dishName,
    currentStepIndex: job.currentStepIndex,
    totalSteps: job.totalSteps,
    stitchedVideoUrl: job.stitchedVideoUrl,
    error: job.error,
    steps: job.steps.map((st) => ({
      stepNumber: st.stepNumber,
      actionTitle: st.actionTitle,
      station: st.station,
      voiceoverText: st.voiceoverText,
      imageUrl: st.imageUrl,
      videoUrl: st.videoUrl,
      status: st.status,
    })),
    updatedAt: job.updatedAt,
  });
});

// REST ENDPOINT: 3. List recent jobs
app.get('/api/recipes/pipeline/jobs', (_req, res) => {
  const list = Array.from(recipeJobs.values())
    .map((j) => ({
      jobId: j.id,
      dishName: j.dishName,
      status: j.status,
      progress: j.progress,
      createdAt: j.createdAt,
      stitchedVideoUrl: j.stitchedVideoUrl,
    }))
    .reverse();

  return res.json({ jobs: list.slice(0, 20) });
});

// ==========================================
// ELEVENLABS PROXY & SYNTHESIS ENDPOINTS
// ==========================================
function getElevenLabsApiKey(req: express.Request): string | null {
  const customHeader = req.headers['x-elevenlabs-key'] as string;
  if (customHeader && customHeader.trim()) {
    return customHeader.trim();
  }
  return process.env.ELEVENLABS_API_KEY || null;
}

// Check ElevenLabs user account status & remaining quota
app.get('/api/elevenlabs/user', async (req, res) => {
  try {
    const apiKey = getElevenLabsApiKey(req);
    if (!apiKey) {
      return res.status(400).json({ error: 'Nenhuma chave da ElevenLabs fornecida.' });
    }

    const response = await fetch('https://api.elevenlabs.io/v1/user', {
      headers: {
        'xi-api-key': apiKey,
      },
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({
        error: `Falha na verificação da ElevenLabs: ${errText || response.statusText}`,
      });
    }

    const data = await response.json();
    const sub = data.subscription || {};
    return res.json({
      tier: sub.tier || 'free',
      character_count: sub.character_count || 0,
      character_limit: sub.character_limit || 10000,
    });
  } catch (error: any) {
    return res.status(500).json({ error: error.message || 'Erro ao consultar ElevenLabs' });
  }
});

// List ElevenLabs voices
app.get('/api/elevenlabs/voices', async (req, res) => {
  try {
    const apiKey = getElevenLabsApiKey(req);
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers['xi-api-key'] = apiKey;
    }

    const response = await fetch('https://api.elevenlabs.io/v1/voices', { headers });
    if (!response.ok) {
      // Return curated standard voices as fallback
      return res.json({
        voices: [
          { voice_id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel', category: 'premade', labels: { accent: 'Natural Gastronômica' } },
          { voice_id: 'AZnzlk1XvdvUeBnXmlld', name: 'Domi', category: 'premade', labels: { accent: 'Confiante' } },
          { voice_id: 'EXAVITQu4vr4xnSDxMaL', name: 'Bella', category: 'premade', labels: { accent: 'Expressiva' } },
          { voice_id: 'ErXwobaYiN019PkySvjV', name: 'Antoni', category: 'premade', labels: { accent: 'Narrativa' } },
        ],
      });
    }

    const data = await response.json();
    return res.json({ voices: data.voices || [] });
  } catch (error: any) {
    return res.json({
      voices: [
        { voice_id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel', category: 'premade' },
        { voice_id: 'AZnzlk1XvdvUeBnXmlld', name: 'Domi', category: 'premade' },
      ],
    });
  }
});

// Synthesize speech via ElevenLabs with audio buffer return
app.post('/api/elevenlabs/tts', async (req, res) => {
  try {
    const { text, voiceId = '21m00Tcm4TlvDq8ikWAM' } = req.body;
    const apiKey = getElevenLabsApiKey(req);

    if (!apiKey) {
      return res.status(400).json({ error: 'Chave da ElevenLabs não configurada' });
    }
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Texto para locução é obrigatório' });
    }

    const cleanText = text.trim();
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=mp3_44100_128`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': apiKey,
          'Content-Type': 'application/json',
          Accept: 'audio/mpeg',
        },
        body: JSON.stringify({
          text: cleanText,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.8,
            style: 0.2,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({
        error: `Erro ElevenLabs: ${errText || response.statusText}`,
      });
    }

    const arrayBuf = await response.arrayBuffer();
    const base64Audio = Buffer.from(arrayBuf).toString('base64');

    return res.json({
      audio: base64Audio,
      mimeType: 'audio/mp3',
      provider: 'elevenlabs',
    });
  } catch (error: any) {
    return res.status(500).json({ error: error.message || 'Falha na sintetização ElevenLabs' });
  }
});

// 7. Unified High-Fidelity Text-to-Speech (TTS) for Voiceover Narration:
// Priority: 1. ElevenLabs (if key provided) -> 2. Gemini 3.8 Flash TTS -> 3. Local browser Web Speech API
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'Kore', voiceId } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Texto para locução é obrigatório.' });
    }

    const cleanText = text.trim();
    const elevenKey = getElevenLabsApiKey(req);

    // ==========================================
    // 1. ELEVENLABS ATTEMPT (SE CHAVE ESTIVER DISPONÍVEL)
    // ==========================================
    if (elevenKey) {
      try {
        const vId = voiceId || '21m00Tcm4TlvDq8ikWAM';
        const elevenRes = await fetch(
          `https://api.elevenlabs.io/v1/text-to-speech/${vId}?output_format=mp3_44100_128`,
          {
            method: 'POST',
            headers: {
              'xi-api-key': elevenKey,
              'Content-Type': 'application/json',
              Accept: 'audio/mpeg',
            },
            body: JSON.stringify({
              text: cleanText,
              model_id: 'eleven_multilingual_v2',
              voice_settings: {
                stability: 0.5,
                similarity_boost: 0.8,
              },
            }),
          }
        );

        if (elevenRes.ok) {
          const arrayBuf = await elevenRes.arrayBuffer();
          const base64Audio = Buffer.from(arrayBuf).toString('base64');
          return res.json({
            available: true,
            audio: base64Audio,
            mimeType: 'audio/mp3',
            provider: 'elevenlabs',
          });
        }
        // Se a cota da ElevenLabs acabar (401, 429), prossegue suavemente para o fallback do Gemini!
      } catch (e) {
        console.warn('ElevenLabs falhou, utilizando fallback automático Gemini TTS:', e);
      }
    }

    // ==========================================
    // 2. FALLBACK PARA GEMINI 3.8 FLASH TTS
    // ==========================================
    const voiceName = voice || 'Kore';
    const cacheKey = `${voiceName}:${cleanText}`;

    // Serve from in-memory cache if already generated
    const cached = ttsCache.get(cacheKey);
    if (cached) {
      return res.json({
        available: true,
        cached: true,
        audio: cached.audio,
        mimeType: cached.mimeType,
        sampleRate: cached.sampleRate,
        provider: 'gemini',
      });
    }

    if (ai) {
      // Primary model: gemini-3.8-flash-tts, Fallback: gemini-3.8-flash-lite-tts
      const candidateModels = ['gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts'];
      let lastErrText = '';
      let isQuotaExceeded = false;

      for (const model of candidateModels) {
        try {
          const response = await ai.models.generateContent({
            model,
            contents: [
              {
                role: 'user',
                parts: [
                  {
                    text: cleanText,
                  },
                ],
              },
            ],
            config: {
              responseModalities: ['AUDIO'],
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: voiceName },
                },
              },
            },
          });

          const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
          if (base64Audio) {
            const resultData = {
              audio: base64Audio,
              mimeType: 'audio/pcm;rate=24000',
              sampleRate: 24000,
            };

            // Store in cache (keep max 120 items)
            if (ttsCache.size > 120) {
              const firstKey = ttsCache.keys().next().value;
              if (firstKey) ttsCache.delete(firstKey);
            }
            ttsCache.set(cacheKey, resultData);

            return res.json({
              available: true,
              ...resultData,
            });
          }
        } catch (modelErr: any) {
          lastErrText = String(modelErr?.message || modelErr || '');
          if (
            lastErrText.includes('429') ||
            lastErrText.includes('RESOURCE_EXHAUSTED') ||
            lastErrText.includes('Quota exceeded') ||
            lastErrText.includes('rate-limit')
          ) {
            isQuotaExceeded = true;
          }
          // Continue loop to try fallback model
        }
      }

      // If models hit rate limit or quota, respond gracefully without throwing or crashing
      if (isQuotaExceeded) {
        return res.json({
          available: false,
          quotaExceeded: true,
          message: 'Limite temporário de cota do sintetizador de voz Gemini atingido. O vídeo continuará com a trilha sonora e efeitos culinários integrados.',
        });
      }
    }

    return res.json({
      available: false,
      quotaExceeded: false,
      message: 'Sintetizador de áudio online indisponível.',
    });
  } catch (error: any) {
    return res.status(200).json({
      available: false,
      quotaExceeded: false,
      message: 'Falha na geração de locução TTS. Mantendo trilha sonora integrada.',
    });
  }
});

// Helper to synthesize speech directly to an MP3 file on the server
async function generateSpeechAudioFile(
  text: string,
  outputPath: string,
  tempDir: string,
  req?: express.Request
): Promise<boolean> {
  const cleanText = cleanAndPolishPortugueseCaption(text);
  if (!cleanText) return false;

  // 1. ElevenLabs if key available
  const elevenKey = req ? getElevenLabsApiKey(req) : process.env.ELEVENLABS_API_KEY;
  if (elevenKey) {
    try {
      const vId = '21m00Tcm4TlvDq8ikWAM';
      const elevenRes = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${vId}?output_format=mp3_44100_128`,
        {
          method: 'POST',
          headers: {
            'xi-api-key': elevenKey,
            'Content-Type': 'application/json',
            Accept: 'audio/mpeg',
          },
          body: JSON.stringify({
            text: cleanText,
            model_id: 'eleven_multilingual_v2',
            voice_settings: { stability: 0.5, similarity_boost: 0.8 },
          }),
        }
      );
      if (elevenRes.ok) {
        const arrayBuf = await elevenRes.arrayBuffer();
        fs.writeFileSync(outputPath, Buffer.from(arrayBuf));
        return true;
      }
    } catch (_) {}
  }

  // 2. Gemini 3.8 Flash TTS fallback
  if (ai) {
    const candidateModels = ['gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts'];
    for (const model of candidateModels) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: [{ role: 'user', parts: [{ text: cleanText }] }],
          config: {
            responseModalities: ['AUDIO'],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: 'Kore' },
              },
            },
          },
        });
        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (base64Audio) {
          const rawBuffer = Buffer.from(base64Audio, 'base64');
          const pcmPath = path.join(tempDir, `tts_pcm_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.raw`);
          fs.writeFileSync(pcmPath, rawBuffer);
          await new Promise<void>((resolve) => {
            const p = spawn('ffmpeg', [
              '-y',
              '-f', 's16le',
              '-ar', '24000',
              '-ac', '1',
              '-i', pcmPath,
              '-c:a', 'libmp3lame',
              '-b:a', '192k',
              outputPath,
            ]);
            p.on('close', () => {
              try { fs.unlinkSync(pcmPath); } catch (_) {}
              resolve();
            });
            p.on('error', () => resolve());
          });
          if (fs.existsSync(outputPath) && fs.statSync(outputPath).size > 100) {
            return true;
          }
        }
      } catch (err) {
        console.warn(`TTS attempt with ${model} error:`, err);
      }
    }
  }

  return false;
}

// ==========================================
// 8. FFMPEG SPEED RAMP, ZERO-AUDIO CLIPS & UNIFIED AUDIO MASTER PIPELINE
// Implements:
// 1. -an on all individual step clips (100% silent intermediate clips)
// 2. Unified master narration audio (audio_master.mp3) with zero speech overlap
// 3. Audio-driven sync: cut each video strictly to step voice duration
// 4. -g 30 keyframe alignment and strict .mp4 (H.264/AAC) output (Zero WebM freezing)
// ==========================================

app.post('/api/ffmpeg/process-recipe', async (req, res) => {
  const tempDir = path.join(os.tmpdir(), `cineforge_rec_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`);
  try {
    const {
      steps = [],
      speedFactor = 1.0,
      trimStartSeconds = 0,
      burnSubtitles = true,
      audioDrivenSync = false,
      enableMusic = true,
      preserveAiVideoDuration = true,
      preserveVideoAudio = true,
    } = req.body;

    if (!Array.isArray(steps) || steps.length === 0) {
      return res.status(400).json({ error: 'Nenhuma etapa de vídeo fornecida para o FFmpeg processar.' });
    }

    // Verify ffmpeg & ffprobe binary
    const hasFfmpeg = await new Promise<boolean>((resolve) => {
      const proc = spawn('ffmpeg', ['-version']);
      proc.on('error', () => resolve(false));
      proc.on('close', (code) => resolve(code === 0));
    });

    if (!hasFfmpeg) {
      return res.status(503).json({ error: 'FFmpeg não disponível no servidor host.' });
    }

    fs.mkdirSync(tempDir, { recursive: true });

    const ptsMultiplier = Math.max(0.5, Math.min(1.0, 1.0 / (speedFactor || 1.0)));
    const fontPath = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf';
    const hasFont = fs.existsSync(fontPath);

    const stepAudioFiles: string[] = [];
    const stepDurations: number[] = [];

    // ==========================================
    // ETAPA 1 & 2: GERAR OU PROCESSAR ÁUDIO DA LOCUÇÃO DE CADA PASSO
    // ==========================================
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const audioOutPath = path.join(tempDir, `step_${i}_audio.mp3`);
      let audioReady = false;

      // Se o cliente enviou o áudio sintetizado em base64 (Gemini / ElevenLabs)
      if (step.audioBase64) {
        try {
          const rawBuffer = Buffer.from(step.audioBase64, 'base64');
          if (step.mimeType?.includes('pcm')) {
            const pcmPath = path.join(tempDir, `step_${i}_pcm.raw`);
            fs.writeFileSync(pcmPath, rawBuffer);
            await new Promise<void>((resolve) => {
              const p = spawn('ffmpeg', [
                '-y',
                '-f', 's16le',
                '-ar', '24000',
                '-ac', '1',
                '-i', pcmPath,
                '-c:a', 'libmp3lame',
                '-b:a', '192k',
                audioOutPath,
              ]);
              p.on('close', () => resolve());
            });
            audioReady = fs.existsSync(audioOutPath);
          } else {
            fs.writeFileSync(audioOutPath, rawBuffer);
            audioReady = true;
          }
        } catch (_) {}
      }

      // Se o cliente não enviou áudio em base64, sintetiza automaticamente a locução com Gemini TTS
      if (!audioReady && (step.voiceoverText || step.instruction || step.actionTitle)) {
        const textToSpeak = step.voiceoverText || step.instruction || step.actionTitle;
        audioReady = await generateSpeechAudioFile(textToSpeak, audioOutPath, tempDir, req);
      }

      // Se não temos áudio de voz para este passo, gerar silêncio
      if (!audioReady) {
        const fallbackDuration = Math.max(2.0, (step.durationSeconds || 5) / (speedFactor || 1.0));
        await new Promise<void>((resolve) => {
          const p = spawn('ffmpeg', [
            '-y',
            '-f', 'lavfi',
            '-i', `anullsrc=r=44100:cl=stereo`,
            '-t', fallbackDuration.toFixed(2),
            '-c:a', 'libmp3lame',
            '-b:a', '128k',
            audioOutPath,
          ]);
          p.on('close', () => resolve());
        });
      }

      // Medir duração do áudio da locução com ffprobe
      let voiceDurationSec = Math.max(2.0, (step.durationSeconds || 5) / (speedFactor || 1.0));
      try {
        const probeProc = spawn('ffprobe', [
          '-v', 'error',
          '-show_entries', 'format=duration',
          '-of', 'default=noprint_wrappers=1:nokey=1',
          audioOutPath,
        ]);
        let probeOut = '';
        probeProc.stdout.on('data', (d) => (probeOut += d.toString()));
        await new Promise((resolve) => probeProc.on('close', resolve));
        const parsedDur = parseFloat(probeOut.trim());
        if (!isNaN(parsedDur) && parsedDur > 0.5) {
          voiceDurationSec = parsedDur;
        }
      } catch (_) {}

      stepDurations.push(voiceDurationSec);
      stepAudioFiles.push(audioOutPath);
    }

    // ==========================================
    // ETAPA 3: PROCESSAR CADA VÍDEO PRESERVANDO O VÍDEO COMPLETO E O ÁUDIO NATIVO (SFX)
    // ==========================================
    const processedClipPaths: string[] = [];
    let anyClipHasNativeAudio = false;

    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const videoSrc = step.videoUrl || step.imageUrl;
      if (!videoSrc) continue;

      const inputExt = videoSrc.includes('.mp4') ? 'mp4' : videoSrc.includes('.webm') ? 'webm' : 'jpg';
      const rawInputPath = path.join(tempDir, `raw_step_${i}.${inputExt}`);
      const stepOutPath = path.join(tempDir, `processed_step_${i}.mp4`);

      // Download ou cópia do asset
      if (videoSrc.startsWith('http')) {
        const resp = await fetch(videoSrc);
        if (!resp.ok) continue;
        const buf = await resp.arrayBuffer();
        fs.writeFileSync(rawInputPath, Buffer.from(buf));
      } else if (fs.existsSync(path.resolve(__dirname, videoSrc.replace(/^\//, '')))) {
        fs.copyFileSync(path.resolve(__dirname, videoSrc.replace(/^\//, '')), rawInputPath);
      } else {
        continue;
      }

      const isVideo = inputExt === 'mp4' || inputExt === 'webm';

      // 1. Detectar duração real do vídeo original da IA
      let nativeVideoDuration = 0;
      let hasNativeAudio = false;

      if (isVideo) {
        try {
          const probe = spawn('ffprobe', [
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            rawInputPath,
          ]);
          let out = '';
          probe.stdout.on('data', (d) => (out += d.toString()));
          await new Promise((r) => probe.on('close', r));
          nativeVideoDuration = parseFloat(out.trim()) || 0;
        } catch (_) {}

        // Verificar se tem áudio nativo (chiado, corte, fritura)
        try {
          const aProbe = spawn('ffprobe', [
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            rawInputPath,
          ]);
          let aOut = '';
          aProbe.stdout.on('data', (d) => (aOut += d.toString()));
          await new Promise((r) => aProbe.on('close', r));
          hasNativeAudio = aOut.includes('audio');
          if (hasNativeAudio) anyClipHasNativeAudio = true;
        } catch (_) {}
      }

      // Determinar duração do clipe:
      // Se preserveAiVideoDuration = true e temos a duração do vídeo da Agnes, usamos o vídeo TODO sem cortar!
      let clipDuration: number;
      if (isVideo && nativeVideoDuration > 1.0 && preserveAiVideoDuration) {
        clipDuration = nativeVideoDuration;
      } else if (audioDrivenSync && stepDurations[i]) {
        clipDuration = stepDurations[i] + 0.2;
      } else {
        clipDuration = Math.max(2.0, (step.durationSeconds || 5) / (speedFactor || 1.0));
      }

      // Ajustar o áudio da locução deste passo para preencher a duração exata do clipe com silêncio suave no final
      const stepAudioPath = stepAudioFiles[i];
      if (fs.existsSync(stepAudioPath) && clipDuration > (stepDurations[i] || 0)) {
        const paddedAudioPath = path.join(tempDir, `step_${i}_audio_padded.mp3`);
        const silenceNeeded = clipDuration - (stepDurations[i] || 0);
        await new Promise<void>((resolve) => {
          const p = spawn('ffmpeg', [
            '-y',
            '-i', stepAudioPath,
            '-af', `apad=pad_dur=${silenceNeeded.toFixed(2)}`,
            '-t', clipDuration.toFixed(2),
            '-c:a', 'libmp3lame',
            '-b:a', '192k',
            paddedAudioPath,
          ]);
          p.on('close', () => {
            if (fs.existsSync(paddedAudioPath)) {
              stepAudioFiles[i] = paddedAudioPath;
            }
            resolve();
          });
          p.on('error', () => resolve());
        });
      }

      // Preparar arquivo de legenda UTF-8 para drawtext
      const rawCaption = (step.voiceoverText || step.actionTitle || `Tomada ${i + 1}`)
        .replace(/\n/g, ' ')
        .trim();
      const safeCaption = cleanAndPolishPortugueseCaption(rawCaption);

      const captionFile = path.join(tempDir, `step_${i}_caption.txt`);
      if (burnSubtitles && safeCaption) {
        // Quebra em linhas curtas de 32 chars
        const words = safeCaption.split(' ');
        let l1 = '', l2 = '';
        for (const w of words) {
          if ((l1 + ' ' + w).length < 28) l1 = (l1 ? l1 + ' ' : '') + w;
          else l2 = (l2 ? l2 + ' ' : '') + w;
        }
        const textToBurn = l2 ? `${l1}\n${l2}` : l1;
        fs.writeFileSync(captionFile, textToBurn, 'utf8');
      }

      await new Promise<void>((resolve) => {
        let filterParts: string[] = [];

        if (isVideo && !preserveAiVideoDuration && trimStartSeconds > 0) {
          filterParts.push(`trim=start=${trimStartSeconds}`);
          filterParts.push(`setpts=${ptsMultiplier.toFixed(2)}*PTS`);
        }
        filterParts.push(`scale=720:1280:force_original_aspect_ratio=increase`);
        filterParts.push(`crop=720:1280`);

        // Queima de legendas amarelas com alto contraste (Reels/TikTok) usando textfile
        if (burnSubtitles && hasFont && fs.existsSync(captionFile)) {
          filterParts.push(
            `drawtext=fontfile=${fontPath}:textfile='${captionFile}':fontsize=32:fontcolor=yellow:borderw=3:bordercolor=black:box=1:boxcolor=black@0.75:boxborderw=14:line_spacing=10:x=(w-text_w)/2:y=h*0.74`
          );
        }

        const vf = filterParts.join(',');

        let args: string[] = [];

        if (isVideo) {
          if (hasNativeAudio && preserveVideoAudio) {
            // Preserva o áudio do vídeo original (SFX da fritura/corte) normalizado
            args = [
              '-y',
              '-i', rawInputPath,
              '-vf', vf,
              '-t', clipDuration.toFixed(2),
              '-c:v', 'libx264',
              '-preset', 'veryfast',
              '-pix_fmt', 'yuv420p',
              '-g', '30',
              '-c:a', 'aac',
              '-ar', '44100',
              '-ac', '2',
              '-b:a', '192k',
              stepOutPath,
            ];
          } else {
            // Se o vídeo não possui áudio nativo, gera silêncio sincronizado para que todos os clipes tenham trilha de áudio
            args = [
              '-y',
              '-i', rawInputPath,
              '-f', 'lavfi',
              '-i', `anullsrc=r=44100:cl=stereo`,
              '-vf', vf,
              '-t', clipDuration.toFixed(2),
              '-c:v', 'libx264',
              '-preset', 'veryfast',
              '-pix_fmt', 'yuv420p',
              '-g', '30',
              '-c:a', 'aac',
              '-shortest',
              stepOutPath,
            ];
          }
        } else {
          // Imagem estática: faz loop com silêncio
          args = [
            '-y',
            '-loop', '1',
            '-i', rawInputPath,
            '-f', 'lavfi',
            '-i', `anullsrc=r=44100:cl=stereo`,
            '-t', clipDuration.toFixed(2),
            '-vf', `${vf},format=yuv420p`,
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p',
            '-g', '30',
            '-c:a', 'aac',
            '-shortest',
            stepOutPath,
          ];
        }

        const proc = spawn('ffmpeg', args);
        proc.on('close', (code) => {
          if (code === 0 && fs.existsSync(stepOutPath)) {
            processedClipPaths.push(stepOutPath);
          }
          resolve();
        });
        proc.on('error', () => resolve());
      });
    }

    if (processedClipPaths.length === 0) {
      return res.status(400).json({ error: 'Nenhum clipe pôde ser processado pelo FFmpeg.' });
    }

    // ==========================================
    // ETAPA 4: CONCATENAR VÍDEOS COM SUA TRILHA DE ÁUDIO NATIVO (SFX)
    // ==========================================
    const videoListPath = path.join(tempDir, 'video_list.txt');
    fs.writeFileSync(videoListPath, processedClipPaths.map((p) => `file '${p}'`).join('\n'));

    const videoConcatPath = path.join(tempDir, 'video_concat.mp4');

    await new Promise<void>((resolve, reject) => {
      const p = spawn('ffmpeg', [
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', videoListPath,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-g', '30',
        '-c:a', 'aac',
        '-ar', '44100',
        '-ac', '2',
        videoConcatPath,
      ]);
      p.on('close', (code) => {
        if (code === 0 && fs.existsSync(videoConcatPath)) resolve();
        else reject(new Error(`Falha ao concatenar vídeos (code ${code})`));
      });
      p.on('error', reject);
    });

    // ==========================================
    // ETAPA 5: CONCATENAR ÁUDIO MASTER UNIFICADO (ZERO ENCAVALAMENTO DE VOZ)
    // ==========================================
    const audioListPath = path.join(tempDir, 'audio_list.txt');
    fs.writeFileSync(audioListPath, stepAudioFiles.map((p) => `file '${p}'`).join('\n'));

    const audioMasterPath = path.join(tempDir, 'audio_master.mp3');

    await new Promise<void>((resolve, reject) => {
      const p = spawn('ffmpeg', [
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', audioListPath,
        '-c:a', 'libmp3lame',
        '-b:a', '192k',
        audioMasterPath,
      ]);
      p.on('close', (code) => {
        if (code === 0 && fs.existsSync(audioMasterPath)) resolve();
        else reject(new Error(`Falha ao concatenar áudio master (code ${code})`));
      });
      p.on('error', reject);
    });

    // ==========================================
    // ETAPA 6: MUX FINAL MP4 (H.264 / AAC)
    // Mixa o som nativo da IA (fritura, corte a 0.65 de volume) com a narração limpa (1.0 de volume)
    // ==========================================
    const finalOutputPath = path.join(tempDir, 'video_final_recipe.mp4');

    await new Promise<void>((resolve, reject) => {
      let args: string[];

      if (anyClipHasNativeAudio && preserveVideoAudio) {
        // Mixa o áudio nativo do vídeo (SFX) com a locução master
        args = [
          '-y',
          '-i', videoConcatPath,
          '-i', audioMasterPath,
          '-filter_complex', '[0:a]volume=0.65[sfx];[1:a]volume=1.0[voice];[sfx][voice]amix=inputs=2:duration=first:dropout_transition=2[aout]',
          '-map', '0:v',
          '-map', '[aout]',
          '-c:v', 'copy',
          '-c:a', 'aac',
          '-b:a', '192k',
          finalOutputPath,
        ];
      } else {
        // Usa a locução master diretamente sobre o vídeo
        args = [
          '-y',
          '-i', videoConcatPath,
          '-i', audioMasterPath,
          '-c:v', 'copy',
          '-c:a', 'aac',
          '-b:a', '192k',
          '-shortest',
          finalOutputPath,
        ];
      }

      const p = spawn('ffmpeg', args);
      p.on('close', (code) => {
        if (code === 0 && fs.existsSync(finalOutputPath)) resolve();
        else reject(new Error(`Falha na multiplexação final MP4 (code ${code})`));
      });
      p.on('error', reject);
    });

    // Transmissão do MP4 final de alta compatibilidade para download/visualização
    const stat = fs.statSync(finalOutputPath);
    res.writeHead(200, {
      'Content-Type': 'video/mp4',
      'Content-Length': stat.size,
      'Content-Disposition': 'inline; filename="receita_final_fluida_agnes.mp4"',
    });

    const readStream = fs.createReadStream(finalOutputPath);
    readStream.pipe(res);
    readStream.on('close', () => {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch (_) {}
    });
  } catch (err: any) {
    console.error('Error in /api/ffmpeg/process-recipe:', err);
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (_) {}
    return res.status(500).json({ error: err.message || 'Erro ao processar receita no FFmpeg' });
  }
});

// 9. CRITICAL: 404 handler for all /api/* routes so unmatched calls return JSON instead of Vite index.html
app.all('/api/*', (req, res) => {
  return res.status(404).json({
    error: `Endpoint de API não encontrado: ${req.method} ${req.path}`,
    status: 404,
  });
});

// Setup Vite middleware in dev or static files in prod
async function startServer() {
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, HOST, () => {
    console.log(`Server listening on http://${HOST}:${PORT} (dev: ${isDev})`);
  });
}

startServer();
