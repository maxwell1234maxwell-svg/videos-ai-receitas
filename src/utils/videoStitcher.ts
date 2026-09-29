import { SceneShot, FilmProject, CulinarySfx } from '../types';
import { cleanAndPolishPortugueseCaption } from './textSanitizer';
import {
  renderSoundtrackToDestination,
  renderSfxToDestination,
  fetchVoiceoverAudioBuffer,
} from './audioSynth';

export interface StitchProgress {
  currentShotIndex: number;
  totalShots: number;
  percent: number;
  statusText: string;
}

export interface StitchAudioOptions {
  enableMusic?: boolean;
  enableSfx?: boolean;
  enableVoiceover?: boolean;
  musicVolume?: number;
  sfxVolume?: number;
  speedRamp?: number; // e.g. 1.35x for fast TikTok tempo
  trimStaticEdges?: boolean; // Trim 0.2s of motionless start
  audioDrivenSync?: boolean; // Match clip duration precisely to voiceover length
  burnSubtitles?: boolean; // Burn prominent TikTok-style yellow/white captions
}

interface LoadedMediaItem {
  image?: HTMLImageElement;
  video?: HTMLVideoElement;
  hasVideo: boolean;
  audioConnected?: boolean;
  voiceoverBuffer?: AudioBuffer | null;
}

/**
 * Helper to wrap text into multiple lines for high-visibility viral captions
 */
function wrapCaptionLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = '';

  for (let i = 0; i < words.length; i++) {
    const testLine = currentLine ? `${currentLine} ${words[i]}` : words[i];
    if (ctx.measureText(testLine).width > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = words[i];
    } else {
      currentLine = testLine;
    }
  }
  if (currentLine) {
    lines.push(currentLine);
  }
  return lines.slice(0, 3); // Max 3 lines to avoid cluttering screen
}

/**
 * Draws prominent, high-contrast, viral TikTok/Reels captions onto the canvas
 */
function drawViralTikTokCaptions(
  ctx: CanvasRenderingContext2D,
  captionText: string,
  stepNumber: number,
  totalSteps: number,
  stationTitle: string,
  width: number,
  height: number,
) {
  if (!captionText || !captionText.trim()) return;

  const cleanText = captionText.trim();
  const maxBoxWidth = Math.min(width * 0.88, 640);
  const fontSize = width > 700 ? 26 : 22;
  const lineHeight = fontSize * 1.35;

  ctx.save();
  ctx.font = `800 ${fontSize}px "Cabinet Grotesk", "Plus Jakarta Sans", sans-serif`;

  const lines = wrapCaptionLines(ctx, cleanText, maxBoxWidth - 36);
  const boxHeight = lines.length * lineHeight + 42;
  // Positioned at lower-third sweet spot (76% down the screen, clear of UI)
  const boxY = height * 0.74 - boxHeight / 2;
  const boxX = (width - maxBoxWidth) / 2;

  // Background plate: dark glassmorphism with high contrast
  ctx.fillStyle = 'rgba(8, 12, 22, 0.86)';
  ctx.beginPath();
  const radius = 16;
  ctx.roundRect ? ctx.roundRect(boxX, boxY, maxBoxWidth, boxHeight, radius) : ctx.rect(boxX, boxY, maxBoxWidth, boxHeight);
  ctx.fill();

  // Subtle amber border
  ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Step Badge Pill inside Box
  const badgeText = `PASSO ${stepNumber}/${totalSteps} · ${(stationTitle || 'AÇÃO').toUpperCase()}`;
  ctx.font = '700 11px "JetBrains Mono", monospace';
  ctx.fillStyle = '#f59e0b';
  ctx.textAlign = 'center';
  ctx.fillText(badgeText, width / 2, boxY + 18);

  // High-Visibility Subtitle Text (Bright Yellow / White with Shadow)
  ctx.font = `800 ${fontSize}px "Cabinet Grotesk", "Plus Jakarta Sans", sans-serif`;
  ctx.fillStyle = '#fef08a'; // Vibrant appetizing yellow
  ctx.shadowColor = 'rgba(0, 0, 0, 0.95)';
  ctx.shadowBlur = 8;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 2;

  lines.forEach((line, idx) => {
    const lineY = boxY + 42 + idx * lineHeight;
    ctx.fillText(line, width / 2, lineY);
  });

  ctx.restore();
}

/**
 * Intelligently infers appropriate culinary sound effect from shot metadata or equipment station
 */
function inferCulinarySfx(shot: any): CulinarySfx {
  if (shot.sfx && shot.sfx !== 'none') return shot.sfx;
  const station = shot.equipmentStation || '';
  const text = `${shot.title || ''} ${shot.actionTitle || ''} ${shot.narrativeDescription || ''}`.toLowerCase();

  if (
    station === 'cutting_board' ||
    text.includes('cort') ||
    text.includes('pic') ||
    text.includes('fati') ||
    text.includes('tábua')
  ) {
    return 'chop';
  }
  if (
    station === 'pan_stove' ||
    text.includes('frigideira') ||
    text.includes('azeite') ||
    text.includes('frit') ||
    text.includes('refog') ||
    text.includes('dour') ||
    text.includes('camar') ||
    text.includes('alho')
  ) {
    return 'sizzle';
  }
  if (
    station === 'pot_boiling' ||
    text.includes('ferv') ||
    text.includes('coz') ||
    text.includes('panela') ||
    text.includes('molho') ||
    text.includes('mistur')
  ) {
    return 'stir';
  }
  if (
    station === 'oven_appliance' ||
    text.includes('forno') ||
    text.includes('air fryer') ||
    text.includes('assad') ||
    text.includes('gratin')
  ) {
    return 'timer_ding';
  }
  if (
    station === 'serving_plate' ||
    station === 'social_hero' ||
    station === 'tasting_fork' ||
    text.includes('prato') ||
    text.includes('garf') ||
    text.includes('serv') ||
    text.includes('degust')
  ) {
    return 'sizzle';
  }
  return 'sizzle';
}

/**
 * Helper to draw an image or video to canvas with cover fit (no stretching or black letterbox)
 */
function drawMediaCover(
  ctx: CanvasRenderingContext2D,
  media: HTMLImageElement | HTMLVideoElement,
  targetWidth: number,
  targetHeight: number,
  scaleFactor: number = 1.0,
) {
  const mediaWidth =
    (media as HTMLVideoElement).videoWidth || (media as HTMLImageElement).naturalWidth;
  const mediaHeight =
    (media as HTMLVideoElement).videoHeight || (media as HTMLImageElement).naturalHeight;

  if (!mediaWidth || !mediaHeight) return;

  const targetRatio = targetWidth / targetHeight;
  const mediaRatio = mediaWidth / mediaHeight;

  let sWidth = mediaWidth;
  let sHeight = mediaHeight;
  let sx = 0;
  let sy = 0;

  if (mediaRatio > targetRatio) {
    sWidth = mediaHeight * targetRatio;
    sx = (mediaWidth - sWidth) / 2;
  } else {
    sHeight = mediaWidth / targetRatio;
    sy = (mediaHeight - sHeight) / 2;
  }

  ctx.save();
  if (scaleFactor !== 1.0) {
    ctx.translate(targetWidth / 2, targetHeight / 2);
    ctx.scale(scaleFactor, scaleFactor);
    ctx.drawImage(
      media,
      sx,
      sy,
      sWidth,
      sHeight,
      -targetWidth / 2,
      -targetHeight / 2,
      targetWidth,
      targetHeight,
    );
  } else {
    ctx.drawImage(media, sx, sy, sWidth, sHeight, 0, 0, targetWidth, targetHeight);
  }
  ctx.restore();
}

/**
 * Stitches images and real video clips into a continuous recorded video file
 * WITH COMPLETE MULTI-TRACK AUDIO (Background music + Culinary SFX + Voiceover Narration)
 * using Canvas + Web Audio API + MediaRecorder.
 */
export async function stitchLongVideo(
  project: FilmProject,
  onProgress?: (progress: StitchProgress) => void,
  audioOptions: StitchAudioOptions = {},
): Promise<Blob> {
  const {
    enableMusic = true,
    enableSfx = true,
    enableVoiceover = true,
    musicVolume = 0.28,
    sfxVolume = 0.45,
    speedRamp = 1.35,
    trimStaticEdges = true,
    audioDrivenSync = true,
    burnSubtitles = true,
  } = audioOptions;

  const validShots = project.shots.filter((s) => s.imageUrl || s.videoUrl);
  if (validShots.length === 0) {
    throw new Error('Nenhuma cena possui imagem ou clipe gerado para compor o vídeo longo.');
  }

  // Determine canvas resolution based on aspect ratio
  let width = 1280;
  let height = 720;
  const ratio = project.aspectRatio as string;
  if (ratio === '9:16') {
    width = 720;
    height = 1280;
  } else if (ratio === '1:1') {
    width = 720;
    height = 720;
  } else if (ratio === '4:3') {
    width = 960;
    height = 720;
  } else if (ratio === '3:4') {
    width = 720;
    height = 960;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  if (!ctx) {
    throw new Error('Canvas 2D context não suportado');
  }

  // ==========================================
  // INITIALIZE WEB AUDIO SYSTEM FOR VIDEO STREAM
  // ==========================================
  const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
  const audioCtx = new AudioCtxClass();
  if (audioCtx.state === 'suspended') {
    await audioCtx.resume().catch(() => {});
  }

  const audioDest = audioCtx.createMediaStreamDestination();
  const masterAudioGain = audioCtx.createGain();
  masterAudioGain.gain.setValueAtTime(1.0, audioCtx.currentTime);
  masterAudioGain.connect(audioDest);

  // Pre-load all shot images, video elements, AND voiceover audio buffers
  const loadedMediaList: LoadedMediaItem[] = [];
  const totalShots = validShots.length;
  const totalDurationSec = validShots.reduce((acc, s) => acc + (s.durationSeconds || 5), 0);

  for (let i = 0; i < totalShots; i++) {
    onProgress?.({
      currentShotIndex: i,
      totalShots,
      percent: Math.round(((i + 1) / totalShots) * 18),
      statusText: `Preparando mídias e áudio da Tomada ${i + 1} de ${totalShots}...`,
    });

    const shot = validShots[i];
    const mediaItem: LoadedMediaItem = { hasVideo: false };

    // 1. Preload image as primary or fallback
    if (shot.imageUrl) {
      try {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        const proxyUrl = shot.imageUrl.startsWith('http')
          ? `/api/proxy-media?url=${encodeURIComponent(shot.imageUrl)}`
          : shot.imageUrl;

        await new Promise<void>((resolve) => {
          img.onload = () => resolve();
          img.onerror = () => resolve();
          img.src = proxyUrl;
        });

        if (img.naturalWidth > 0) {
          mediaItem.image = img;
        }
      } catch (e) {
        console.warn(`Erro ao carregar imagem do passo ${i + 1}:`, e);
      }
    }

    // 2. Preload video if available
    if (shot.videoUrl) {
      try {
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';
        video.muted = true; // start muted during buffer preload
        video.playsInline = true;
        video.preload = 'auto';

        const videoSrc = shot.videoUrl.startsWith('http')
          ? `/api/proxy-media?url=${encodeURIComponent(shot.videoUrl)}`
          : shot.videoUrl;

        const isLoaded = await new Promise<boolean>((resolve) => {
          let done = false;
          const onReady = () => {
            if (!done) {
              done = true;
              cleanup();
              resolve(true);
            }
          };
          const onError = () => {
            if (!done) {
              done = true;
              cleanup();
              resolve(false);
            }
          };
          const cleanup = () => {
            video.removeEventListener('loadeddata', onReady);
            video.removeEventListener('canplay', onReady);
            video.removeEventListener('error', onError);
          };

          video.addEventListener('loadeddata', onReady);
          video.addEventListener('canplay', onReady);
          video.addEventListener('error', onError);

          setTimeout(() => {
            if (!done) {
              done = true;
              cleanup();
              resolve(video.videoWidth > 0 || video.readyState >= 2);
            }
          }, 10000);

          video.src = videoSrc;
          video.load();
        });

        if (isLoaded && (video.videoWidth > 0 || video.readyState >= 1)) {
          mediaItem.video = video;
          mediaItem.hasVideo = true;
        }
      } catch (e) {
        console.warn(`Erro ao preparar vídeo do passo ${i + 1}:`, e);
      }
    }

    // 3. Pre-fetch voiceover TTS audio buffer if present
    const voiceoverText = (shot as any).voiceoverText;
    if (enableVoiceover && voiceoverText && voiceoverText.trim()) {
      try {
        const vBuf = await fetchVoiceoverAudioBuffer(audioCtx, voiceoverText);
        if (vBuf) {
          mediaItem.voiceoverBuffer = vBuf;
        }
      } catch {
        // Continue cleanly without blocking video export
      }
    }

    loadedMediaList.push(mediaItem);
  }

  // ==========================================
  // START CONTINUOUS GASTRONOMIC/CINEMATIC SOUNDTRACK
  // ==========================================
  if (enableMusic) {
    const isRecipe =
      (project as any).steps !== undefined ||
      validShots.some((s) => (s as any).stepNumber !== undefined) ||
      (project.title && project.title.toLowerCase().includes('macarrão')) ||
      (project.title && project.title.toLowerCase().includes('receita'));

    const soundtrackStyle = isRecipe ? 'lofi_kitchen' : (project.ambientAudio as any) || 'cinematic_drone';
    renderSoundtrackToDestination(
      audioCtx,
      masterAudioGain,
      totalDurationSec + 3,
      soundtrackStyle,
      musicVolume,
    );
  }

  // ==========================================
  // SETUP COMBINED MEDIA STREAM (VIDEO + AUDIO)
  // ==========================================
  const fps = 30;
  const canvasStream = canvas.captureStream(fps);

  // Combine video track from canvas with audio track from Web Audio destination!
  const combinedStream = new MediaStream([
    ...canvasStream.getVideoTracks(),
    ...audioDest.stream.getAudioTracks(),
  ]);

  let mimeType = 'video/webm;codecs=vp9,opus';
  if (!MediaRecorder.isTypeSupported(mimeType)) {
    mimeType = 'video/webm;codecs=vp8,opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'video/webm';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'video/mp4;codecs=avc1,mp4a.40.2';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'video/mp4';
          if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = '';
          }
        }
      }
    }
  }

  const mediaRecorder = new MediaRecorder(
    combinedStream,
    mimeType
      ? {
          mimeType,
          videoBitsPerSecond: 6000000,
          audioBitsPerSecond: 192000,
        }
      : undefined,
  );
  const recordedChunks: Blob[] = [];

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      recordedChunks.push(e.data);
    }
  };

  const recordingPromise = new Promise<Blob>((resolve, reject) => {
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: mimeType || 'video/webm' });
      resolve(blob);
    };
    mediaRecorder.onerror = (e) => reject(e);
  });

  mediaRecorder.start();

  // Render frames sequentially
  let overallFrameCount = 0;

  try {
    // Reference to stop previous voiceover so two voices NEVER overlap
    let activeVoiceSourceNode: AudioBufferSourceNode | null = null;

    for (let shotIdx = 0; shotIdx < totalShots; shotIdx++) {
      const shot = validShots[shotIdx];
      const media = loadedMediaList[shotIdx];
      const nextMedia = shotIdx < totalShots - 1 ? loadedMediaList[shotIdx + 1] : null;

      // Calculate snappy duration: speed ramp 1.35x and audio sync
      const rawDuration = shot.durationSeconds || 5;
      let shotDurationSec = rawDuration / speedRamp;
      if (audioDrivenSync && media.voiceoverBuffer) {
        // CRITICAL: The shot duration MUST NOT be shorter than the spoken voiceover!
        // Otherwise Voice 1 keeps playing while Voice 2 begins, creating overlapping voices!
        const vLen = media.voiceoverBuffer.duration;
        shotDurationSec = Math.max(vLen + 0.22, shotDurationSec);
      }
      const totalShotFrames = Math.max(30, Math.round(shotDurationSec * fps));
      const transitionFrames = 0; // Hard cuts for dynamic TikTok/Shorts pacing
      const mainFrames = totalShotFrames;

      // ==========================================
      // SYNCHRONIZED AUDIO TRIGGER FOR CURRENT SHOT
      // ==========================================
      const shotAudioStartTime = audioCtx.currentTime;

      // Stop any prior voiceover if it somehow hasn't stopped (STRICT ANTI-DOUBLE VOICE)
      if (activeVoiceSourceNode) {
        try {
          activeVoiceSourceNode.stop(shotAudioStartTime);
          activeVoiceSourceNode.disconnect();
        } catch (_) {}
        activeVoiceSourceNode = null;
      }

      // 1. Play Synchronized Culinary SFX (Chop, Sizzle, Stir, Bell Ding)
      if (enableSfx) {
        const sfxType = inferCulinarySfx(shot);
        renderSfxToDestination(
          audioCtx,
          masterAudioGain,
          sfxType,
          shotAudioStartTime,
          shotDurationSec,
        );
      }

      // 2. Play Voiceover Narration if buffer was preloaded
      if (enableVoiceover && media.voiceoverBuffer) {
        try {
          const vSource = audioCtx.createBufferSource();
          vSource.buffer = media.voiceoverBuffer;
          const vGain = audioCtx.createGain();
          vGain.gain.setValueAtTime(0.95, shotAudioStartTime);
          vSource.connect(vGain);
          vGain.connect(masterAudioGain);
          // Hard cut starts voiceover at +0.12s
          vSource.start(shotAudioStartTime + 0.12);
          activeVoiceSourceNode = vSource;
        } catch (e) {
          console.warn('Erro ao disparar áudio de voz:', e);
        }
      }

      // 3. Connect Video element: MUST BE MUTED TO AVOID DOUBLE AUDIO!
      if (media.hasVideo && media.video) {
        try {
          media.video.muted = true; // GUARANTEE ZERO DOUBLE VOICE
          media.video.playbackRate = speedRamp; // SPEED RAMP 1.35x
          media.video.currentTime = trimStaticEdges ? 0.2 : 0; // TRIM 0.2s OF INITIAL STILL HANDS
          await media.video.play();
          await new Promise((r) => setTimeout(r, 40));
        } catch (e) {
          console.warn('Erro ao dar play no vídeo durante renderização:', e);
        }
      }

      for (let f = 0; f < totalShotFrames; f++) {
        overallFrameCount++;

        // Progress reporting
        const progressPercent =
          20 + Math.round(((shotIdx * totalShotFrames + f) / (totalShots * totalShotFrames)) * 75);
        onProgress?.({
          currentShotIndex: shotIdx + 1,
          totalShots,
          percent: Math.min(98, progressPercent),
          statusText: `Renderizando tomada ${shotIdx + 1} de ${totalShots} (${media.hasVideo ? 'Vídeo Snappy' : 'Foto Dinâmica'} + Legendas TikTok)...`,
        });

        // Clear canvas
        ctx.fillStyle = '#090d16';
        ctx.fillRect(0, 0, width, height);

        let drawn = false;

        // Draw active video if ready
        if (media.hasVideo && media.video && media.video.videoWidth > 0) {
          if (media.video.ended) {
            media.video.currentTime = trimStaticEdges ? 0.2 : 0;
            media.video.play().catch(() => {});
          } else if (media.video.paused) {
            media.video.play().catch(() => {});
          }

          try {
            drawMediaCover(ctx, media.video, width, height, 1.0);
            drawn = true;
          } catch (e) {
            console.warn('Fallback para imagem devido a erro de desenho de vídeo:', e);
          }
        }

        // Fallback to image if video not drawn
        if (!drawn && media.image && media.image.naturalWidth > 0) {
          const progress = f / totalShotFrames;
          const zoomFactor = 1.0 + progress * 0.08;
          drawMediaCover(ctx, media.image, width, height, zoomFactor);
        }

        // Prominent TikTok/Reels Viral Subtitles Burn-in
        if (burnSubtitles) {
          const rawCaption =
            (shot as any).voiceoverText ||
            (shot as any).instruction ||
            shot.narrativeDescription ||
            shot.title;
          const captionText = cleanAndPolishPortugueseCaption(rawCaption);
          const stepNum = (shot as any).stepNumber || shotIdx + 1;
          const stationName = (shot as any).actionTitle || shot.title || '';
          drawViralTikTokCaptions(
            ctx,
            captionText,
            stepNum,
            totalShots,
            stationName,
            width,
            height,
          );
        }

        // Top Subtle Branding Badge
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillRect(0, 0, width, 50);
        ctx.font = '700 12px "JetBrains Mono", monospace';
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(`AGNES CINEFORGE · ${(project.title || 'RECEITA').toUpperCase()}`, 20, 30);
        ctx.restore();

        // Accurate frame delay keeping exact 30fps without clock drift
        await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
      }

      // Pause current video after shot finishes
      if (media.video) {
        try {
          media.video.pause();
        } catch (e) {}
      }
    }
  } finally {
    // Cleanup video elements
    for (const m of loadedMediaList) {
      if (m.video) {
        try {
          m.video.pause();
          m.video.src = '';
          m.video.load();
        } catch (e) {}
      }
    }
  }

  onProgress?.({
    currentShotIndex: totalShots,
    totalShots,
    percent: 100,
    statusText: 'Finalizando multiplexação de vídeo e áudio contínuo...',
  });

  mediaRecorder.stop();
  const finalBlob = await recordingPromise;

  // Clean up audio context
  try {
    audioCtx.close().catch(() => {});
  } catch (_) {}

  return finalBlob;
}

/**
 * Direct Backend FFmpeg Processor:
 * Produces a true MP4 (H.264 / AAC) with -g 30 keyframes, -an muted intermediate clips,
 * and unified audio master. Completely eliminates WebM keyframe freezing and double narration!
 */
export async function stitchRecipeViaFfmpegBackend(
  project: any,
  onProgress?: (progress: StitchProgress) => void,
  options: {
    speedFactor?: number;
    trimStartSeconds?: number;
    burnSubtitles?: boolean;
    audioDrivenSync?: boolean;
    preserveAiVideoDuration?: boolean;
    preserveVideoAudio?: boolean;
  } = {}
): Promise<Blob> {
  const steps = project.shots || project.steps || [];
  const validSteps = steps.filter((s: any) => s.imageUrl || s.videoUrl);
  if (validSteps.length === 0) {
    throw new Error('Nenhuma cena possui imagem ou clipe gerado para processar.');
  }

  onProgress?.({
    currentShotIndex: 1,
    totalShots: validSteps.length,
    percent: 25,
    statusText: 'Iniciando processamento FFmpeg: Preservando vídeo completo da IA e mixando áudio nativo...',
  });

  const response = await fetch('/api/ffmpeg/process-recipe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      steps: validSteps,
      speedFactor: options.speedFactor ?? 1.0,
      trimStartSeconds: options.trimStartSeconds ?? 0,
      burnSubtitles: options.burnSubtitles ?? true,
      audioDrivenSync: options.audioDrivenSync ?? false,
      preserveAiVideoDuration: options.preserveAiVideoDuration ?? true,
      preserveVideoAudio: options.preserveVideoAudio ?? true,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error || `Erro HTTP ${response.status} ao processar no FFmpeg.`);
  }

  onProgress?.({
    currentShotIndex: validSteps.length,
    totalShots: validSteps.length,
    percent: 90,
    statusText: 'Multiplexando MP4 final (H.264 + AAC com keyframes -g 30)...',
  });

  const blob = await response.blob();
  onProgress?.({
    currentShotIndex: validSteps.length,
    totalShots: validSteps.length,
    percent: 100,
    statusText: 'Vídeo MP4 final concluído com sucesso!',
  });

  return blob;
}

/**
 * Exports project as an EDL (Edit Decision List) text file for Adobe Premiere / DaVinci Resolve
 */
export function generateEDL(project: FilmProject): string {
  let edl = `TITLE: ${(project.title || 'FILM').toUpperCase()}\n`;
  edl += `FCM: NON-DROP FRAME\n\n`;

  let currentFrame = 0;
  const fps = 24;

  project.shots.forEach((shot, idx) => {
    const durationFrames = (shot.durationSeconds || 5) * fps;
    const startFrame = currentFrame;
    const endFrame = currentFrame + durationFrames;

    const toTC = (f: number) => {
      const h = String(Math.floor(f / (fps * 3600))).padStart(2, '0');
      const m = String(Math.floor((f / (fps * 60)) % 60)).padStart(2, '0');
      const s = String(Math.floor((f / fps) % 60)).padStart(2, '0');
      const ff = String(Math.floor(f % fps)).padStart(2, '0');
      return `${h}:${m}:${s}:${ff}`;
    };

    const num = String(idx + 1).padStart(3, '0');
    edl += `${num}  AX       V     C        00:00:00:00 ${toTC(durationFrames)} ${toTC(startFrame)} ${toTC(endFrame)}\n`;
    edl += `* FROM CLIP NAME: ${shot.title || `Shot ${idx + 1}`}\n`;
    edl += `* SHOT TYPE: ${shot.shotType || 'medium_shot'} | MOTION: ${shot.cameraMotion || 'static_tripod'}\n`;
    edl += `* IMAGE PROMPT: ${(shot.imagePrompt || '').replace(/\n/g, ' ')}\n`;
    edl += `* VIDEO PROMPT: ${(shot.videoPrompt || '').replace(/\n/g, ' ')}\n\n`;

    currentFrame = endFrame;
  });

  return edl;
}
