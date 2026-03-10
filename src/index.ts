import { AppServer, AppSession, ViewType, AuthenticatedRequest, PhotoData } from '@mentra/sdk';
import { Request, Response } from 'express';
import * as ejs from 'ejs';
import * as path from 'path';
import sharp from 'sharp';

/**
 * Interface representing a stored photo with metadata
 */
interface StoredPhoto {
  requestId: string;
  buffer: Buffer;
  timestamp: Date;
  userId: string;
  mimeType: string;
  filename: string;
  size: number;
}

const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => { throw new Error('PACKAGE_NAME is not set in .env file'); })();
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY ?? (() => { throw new Error('MENTRAOS_API_KEY is not set in .env file'); })();
const PORT = parseInt(process.env.PORT || '3000');
const ALLOW_DEV_VIEW = process.env.ALLOW_DEV_VIEW === 'true';
const SPEAKABOO_API_URL = process.env.SPEAKABOO_API_URL || '';
const SPEAKABOO_API_KEY = process.env.SPEAKABOO_API_KEY || '';
const SPEAKABOO_PROMPT = process.env.SPEAKABOO_PROMPT || 'Analyze this photo.';
const SPEAK_ANALYSIS = process.env.SPEAK_ANALYSIS === 'true';
// Mentra camera configuration (best-effort; ignored if unsupported by device/SDK)
const PHOTO_QUALITY = (process.env.PHOTO_QUALITY || '').toLowerCase() || "small"; // 'small' | 'medium' | 'large'
const PHOTO_MAX_WIDTH = parseInt(process.env.PHOTO_MAX_WIDTH || '0');
const PHOTO_MAX_HEIGHT = parseInt(process.env.PHOTO_MAX_HEIGHT || '0');
const PHOTO_TIMEOUT_MS = parseInt(process.env.PHOTO_TIMEOUT_MS || '0');

/**
 * Photo Taker App with webview functionality for displaying photos
 * Extends AppServer to provide photo taking and webview display capabilities
 */
class ExampleMentraOSApp extends AppServer {
  private photos: Map<string, StoredPhoto> = new Map(); // Store photos by userId
  private latestPhotoTimestamp: Map<string, number> = new Map(); // Track latest photo timestamp per user
  private isStreamingPhotos: Map<string, boolean> = new Map(); // Track if we are streaming photos for a user
  private nextPhotoTime: Map<string, number> = new Map(); // Track next photo time for a user
  private latestAnalysis: Map<string, { requestId: string; answer: string; raw?: any }> = new Map();
  private sessionsByUserId: Map<string, AppSession> = new Map();
  private isSpeaking: Map<string, boolean> = new Map();
  private isSendingToSpeakaboo: Map<string, boolean> = new Map();
  private pendingQuestionTextByUserId: Map<string, string> = new Map();
  private photoCaptureInProgress: Map<string, boolean> = new Map();
  private cameraRequestInFlight: Map<string, boolean> = new Map();
  private transcriptionUnsubscribeByUserId: Map<string, () => void> = new Map();
  private lastFinalTextByUserId: Map<string, string> = new Map();
  private lastFinalAtByUserId: Map<string, number> = new Map();
  private loadingSfxActiveByUserId: Map<string, boolean> = new Map();

  constructor() {
    super({
      packageName: PACKAGE_NAME,
      apiKey: MENTRAOS_API_KEY,
      port: PORT,
    });
    this.setupWebviewRoutes();
  }

  private async requestPhotoMentra(session: AppSession, userId: string) {
    // Build best-effort options; fall back to plain call if unsupported
    const options: any = {};
    if (PHOTO_QUALITY && ['small','medium','large'].includes(PHOTO_QUALITY)) {
      options.size = PHOTO_QUALITY as "small" | "medium" | "large";
    }
    if (PHOTO_MAX_WIDTH > 0) options.maxWidth = PHOTO_MAX_WIDTH;
    if (PHOTO_MAX_HEIGHT > 0) options.maxHeight = PHOTO_MAX_HEIGHT;
    if (PHOTO_TIMEOUT_MS > 0) options.timeoutMs = PHOTO_TIMEOUT_MS;

    try {
      if (Object.keys(options).length > 0) {
        this.logger.info(`Requesting photo with options for ${userId}: ${JSON.stringify(options)}`);
        // @ts-ignore allow SDK to accept options if supported
        return await (session.camera as any).requestPhoto(options);
      }
      return await session.camera.requestPhoto();
    } catch (err) {
      this.logger.warn(`Photo request with options failed, retrying default: ${err}`);
      return await session.camera.requestPhoto();
    }
  }

  private attachTranscription(session: AppSession, userId: string): void {
    // Detach any existing subscription first
    this.detachTranscription(userId);
    const unsubscribe = session.events.onTranscription(async (data: any) => {
      try {
        if (!data.isFinal) {
          // Log interim results just so we can see if the microphone is picking up anything at all
          this.logger.info(`Interim transcription: ${data.text}`);
          return;
        }

        if (this.isSpeaking.get(userId)) {
          this.logger.info(`Skipping transcription handling for user ${userId} because speaking is active.`);
          return;
        }
        
        if (typeof data.text === 'string' && data.text.trim().length > 0) {
          const spoken = data.text.trim();
          this.logger.info(`Final transcription received: "${spoken}"`);
          
          // Only proceed if keyword "ok" or "okay" is present (case-insensitive)
          if (!spoken.toLowerCase().includes('ok') && !spoken.toLowerCase().includes('okay'))  {
            this.logger.info(`Ignoring transcription for ${userId} (no 'ok' or 'okay' keyword)`);
            return;
          }
          // Debounce duplicate finals
          const lastText = this.lastFinalTextByUserId.get(userId) || '';
          const lastAt = this.lastFinalAtByUserId.get(userId) || 0;
          const nowTs = Date.now();
          if (spoken === lastText && nowTs - lastAt < 1500) {
            this.logger.info(`Ignoring duplicate final transcription for ${userId}`);
            return;
          }
          this.lastFinalTextByUserId.set(userId, spoken);
          this.lastFinalAtByUserId.set(userId, nowTs);

          
          // Always capture a fresh photo for a question; ignore cached photos
          this.pendingQuestionTextByUserId.set(userId, spoken);
          if (this.isSendingToSpeakaboo.get(userId)) {
            this.logger.info(`Skipping capture/send for ${userId} because send is in progress; queued text will be used with next photo.`);
            return;
          }
          await this.playLoadingSfx(userId);

          // If a capture is already in progress or camera request is in-flight, let cachePhoto handle sending
          if (this.photoCaptureInProgress.get(userId)) {
            this.logger.info(`Photo capture already in progress for ${userId}; queued question will send with next photo.`);
            return;
          }
          if (this.cameraRequestInFlight.get(userId)) {
            this.logger.info(`Camera request already in flight for ${userId}; queued question will send when photo arrives.`);
            return;
          }

          this.photoCaptureInProgress.set(userId, true);
          try {
            const s = this.sessionsByUserId.get(userId);
            if (!s) {
              this.logger.warn(`No active session for ${userId} to capture photo for queued question.`);
              return;
            }
            this.logger.info(`Capturing photo for queued question for ${userId}`);
            this.cameraRequestInFlight.set(userId, true);
            const newPhoto = await this.requestPhotoMentra(s, userId);
            // Store photo without auto-forwarding default prompt
            await this.cachePhoto(newPhoto, userId, false);
            const photoForSend = this.photos.get(userId) || null;
            const queued = this.pendingQuestionTextByUserId.get(userId);
            if (photoForSend && queued) {
              await this.sendToSpeakaboo(photoForSend, userId, queued);
              this.pendingQuestionTextByUserId.delete(userId);
            }
          } catch (err) {
            this.logger.error(`Queued question capture failed: ${err}`);
            await this.stopAllAudio(userId);
          } finally {
            this.photoCaptureInProgress.set(userId, false);
            this.cameraRequestInFlight.set(userId, false);
          }
        }
      } catch (error) {
        this.logger.error(`Error handling transcription: ${error}`);
      }
    });
    this.transcriptionUnsubscribeByUserId.set(userId, unsubscribe);
  }

  private detachTranscription(userId: string): void {
    const unsub = this.transcriptionUnsubscribeByUserId.get(userId);
    if (unsub) {
      try { unsub(); } catch (_) {}
      this.transcriptionUnsubscribeByUserId.delete(userId);
    }
  }

  private reattachTranscriptionIfNeeded(userId: string): void {
    const session = this.sessionsByUserId.get(userId);
    if (session && !this.transcriptionUnsubscribeByUserId.get(userId)) {
      this.attachTranscription(session, userId);
    }
  }

  private async stopAllAudio(userId: string): Promise<void> {
    const session = this.sessionsByUserId.get(userId);
    if (!session) return;
    try {
      await session.audio.stopAudio();
    } catch (err) {
      this.logger.warn(`stopAudio error for ${userId}: ${err}`);
    }
  }

  private stopLoadingSfx(userId: string): void {
    this.loadingSfxActiveByUserId.set(userId, false);
  }

  private async playLoadingSfx(userId: string): Promise<void> {
    const session = this.sessionsByUserId.get(userId);
    if (!session) return;
    try {
      if (this.loadingSfxActiveByUserId.get(userId)) {
        return;
      }
      // Detach transcription and mark speaking to avoid handling interim transcripts
      this.detachTranscription(userId);
      this.isSpeaking.set(userId, true);

      this.loadingSfxActiveByUserId.set(userId, true);
      const audioUrl = process.env.LOADING_SFX_URL || '';
      (async () => {
        let consecutiveErrors = 0;
        while (this.loadingSfxActiveByUserId.get(userId)) {
          try {
            const result = await (session.audio as any).playAudio({ audioUrl });
            if (!result || !result.success) {
              this.logger.warn(`Loading SFX play failed for ${userId}: ${result?.error || 'Unknown error'}`);
              consecutiveErrors += 1;
              if (consecutiveErrors >= 2) {
                this.logger.warn(`Stopping loading SFX loop for ${userId} due to repeated failures.`);
                this.loadingSfxActiveByUserId.set(userId, false);
                break;
              }
              // Brief backoff on failure to avoid tight loop
              await new Promise((r) => setTimeout(r, 500));
            } else {
              consecutiveErrors = 0;
            }
          } catch (err) {
            this.logger.warn(`Loading SFX play error for ${userId}: ${err}`);
            consecutiveErrors += 1;
            const message = String(err?.message || err || '');
            if (message.includes('WebSocket not connected') || consecutiveErrors >= 2) {
              this.logger.warn(`Stopping loading SFX loop for ${userId} due to repeated/WS errors.`);
              this.loadingSfxActiveByUserId.set(userId, false);
              break;
            }
            // Brief backoff on error to avoid tight loop
            await new Promise((r) => setTimeout(r, 150));
          }
        }
      })();
      this.logger.info(`✅ Loading SFX loop started`);
    }
    catch (error) {
      this.logger.error(`Error starting Loading SFX loop: ${error}`);
    }
  }

  /**
   * Handle new session creation and button press events
   */
  protected async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
    // this gets called whenever a user launches the app
    this.logger.info(`Session started for user ${userId}`);

    // set the initial state of the user
    this.isStreamingPhotos.set(userId, false);
    this.nextPhotoTime.set(userId, Date.now());
    this.sessionsByUserId.set(userId, session);
    this.isSpeaking.set(userId, false);
    this.isSendingToSpeakaboo.set(userId, false);
    this.photoCaptureInProgress.set(userId, false);
    this.cameraRequestInFlight.set(userId, false);

    // this gets called whenever a user presses a button
    session.events.onButtonPress(async (button) => {
      this.logger.info(`Button pressed: ${button.buttonId}, type: ${button.pressType}`);
      // Filter out non-user/system debug events
      if (button.buttonId === 'DEBUG_LOG') {
        return;
      }
      if (button.pressType !== 'short' && button.pressType !== 'long') {
        return;
      }

      if (button.pressType === 'long') {
        // the user held the button, so we toggle the streaming mode
        this.isStreamingPhotos.set(userId, !this.isStreamingPhotos.get(userId));
        this.logger.info(`Streaming photos for user ${userId} is now ${this.isStreamingPhotos.get(userId)}`);
        return;
      } else {
        session.layouts.showTextWall("Button pressed, about to take photo", {durationMs: 4000});
        // the user pressed the button, so we take a single photo
        try {
          // first, get the photo
          if (this.isSendingToSpeakaboo.get(userId) || this.isSpeaking.get(userId)) {
            this.logger.info(`Skipping loading SFX and capture for ${userId} because busy (sending or speaking).`);
          } else {
            await this.playLoadingSfx(userId);
          }
          this.cameraRequestInFlight.set(userId, true);
          const photo = await this.requestPhotoMentra(session, userId);
          // if there was an error, log it
          this.logger.info(`Photo taken for user ${userId}, timestamp: ${photo.timestamp}`);
          this.cachePhoto(photo, userId);
        } catch (error) {
          this.logger.error(`Error taking photo: ${error}`);
          this.stopLoadingSfx(userId);
          await this.stopAllAudio(userId);
        } finally {
          this.cameraRequestInFlight.set(userId, false);
        }
      }
    });

    // attach transcription listener
    this.attachTranscription(session, userId);

    // repeatedly check if we are in streaming mode and if we are ready to take another photo
    setInterval(async () => {
      if (this.isStreamingPhotos.get(userId) && Date.now() > (this.nextPhotoTime.get(userId) ?? 0)) {
        try {
          // set the next photos for 30 seconds from now, as a fallback if this fails
          this.nextPhotoTime.set(userId, Date.now() + 30000);

          // actually take the photo
          this.cameraRequestInFlight.set(userId, true);
          const photo = await this.requestPhotoMentra(session, userId);

          // set the next photo time to now, since we are ready to take another photo
          this.nextPhotoTime.set(userId, Date.now());

          // cache the photo for display
          this.cachePhoto(photo, userId);
        } catch (error) {
          this.logger.error(`Error auto-taking photo: ${error}`);
        } finally {
          this.cameraRequestInFlight.set(userId, false);
        }
      }
    }, 1000);
  }

  protected async onStop(sessionId: string, userId: string, reason: string): Promise<void> {
    // clean up the user's state
    this.isStreamingPhotos.set(userId, false);
    this.nextPhotoTime.delete(userId);
    this.sessionsByUserId.delete(userId);
    this.stopLoadingSfx(userId);
    // detach transcription listener
    this.detachTranscription(userId);
    this.logger.info(`Session stopped for user ${userId}, reason: ${reason}`);
  }

  /**
   * Cache a photo for display
   */
  private async cachePhoto(photo: PhotoData, userId: string, forwardImmediately: boolean = true) {
    // create a new stored photo object which includes the photo data and the user id
    const cachedPhoto: StoredPhoto = {
      requestId: photo.requestId,
      buffer: photo.buffer,
      timestamp: photo.timestamp,
      userId: userId,
      mimeType: photo.mimeType,
      filename: photo.filename,
      size: photo.size
    };

    // this example app simply stores the photo in memory for display in the webview, but you could also send the photo to an AI api,
    // or store it in a database or cloud storage, send it to roboflow, or do other processing here

    // cache the photo for display
    this.photos.set(userId, cachedPhoto);
    // update the latest photo timestamp
    this.latestPhotoTimestamp.set(userId, cachedPhoto.timestamp.getTime());
    this.logger.info(`Photo cached for user ${userId}, timestamp: ${cachedPhoto.timestamp}`);
    

    // If there is a queued question, send photo+text now and skip default forward
    const queued = this.pendingQuestionTextByUserId.get(userId);
    if (queued) {
      this.logger.info(`Found queued question for ${userId}; sending with newly cached photo ${cachedPhoto.requestId}`);
      try {
        await this.sendToSpeakaboo(cachedPhoto, userId, queued);
      } finally {
        this.pendingQuestionTextByUserId.delete(userId);
      }
      return;
    }

    // Fire-and-forget forward to Speakaboo API if configured and immediate forward requested
    if (SPEAKABOO_API_URL && forwardImmediately && !this.isSendingToSpeakaboo.get(userId)) {
      this.forwardPhotoToSpeakaboo(cachedPhoto).catch((err) => {
        this.logger.error(`Error forwarding photo to Speakaboo: ${err}`);
      });
    }
  }

  /**
   * Forward the captured photo to the external Speakaboo API
   */
  private async forwardPhotoToSpeakaboo(photo: StoredPhoto): Promise<void> {
    return this.sendToSpeakaboo(photo, photo.userId, SPEAKABOO_PROMPT);
  }

  /**
   * Shared method to send text (and optional photo) to Speakaboo and handle response/tts
   */
  private async sendToSpeakaboo(photo: StoredPhoto | null, userId: string, promptText: string): Promise<void> {
    if (!SPEAKABOO_API_URL) {
      return;
    }
    try {
      if (this.isSendingToSpeakaboo.get(userId)) {
        this.logger.info(`sendToSpeakaboo skipped for ${userId}: request already in flight.`);
        return;
      }
      this.isSendingToSpeakaboo.set(userId, true);

      const conciseInstruction = 'make sure that the response is concise. Answer in the same language as the question.';
      let promptToSend = promptText || 'Analyze this photo.';
      if (!promptToSend.toLowerCase().includes(conciseInstruction)) {
        promptToSend = `${promptToSend.trim()} ${conciseInstruction}`;
      }
        // Append user-specific custom instructions from settings, if provided
      try {
      const session = this.sessionsByUserId.get(userId);
      const customInstructions = session?.settings.get<string>('custom_instructions', '')?.trim();
      if (customInstructions) {
        promptToSend = `${promptToSend.trim()} ${customInstructions}`;
        }
      } catch (_) {
          // best-effort; ignore if settings unavailable
        }

      let payload: any = { text: promptToSend };
      if (photo) {
        // Compress the image before sending to significantly reduce payload size and upload time
        let imageBuffer = photo.buffer;
        try {
          imageBuffer = await sharp(photo.buffer)
            .resize({ width: 800, withoutEnlargement: true }) // Resize if larger than 800px wide
            .jpeg({ quality: 60 }) // Compress to 60% quality JPEG
            .toBuffer();
          this.logger.info(`Compressed image from ${photo.buffer.length} to ${imageBuffer.length} bytes`);
        } catch (err) {
          this.logger.warn(`Failed to compress image, using original: ${err}`);
        }

        const base64Image = imageBuffer.toString('base64');
        payload = {
          ...payload,
          image: base64Image,
          filename: photo.filename,
          mimeType: photo.mimeType,
          timestamp: photo.timestamp.toISOString(),
        };
      }

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (SPEAKABOO_API_KEY) {
        headers['X-API-KEY'] = SPEAKABOO_API_KEY;
      }

      const startTime = Date.now();
      this.logger.info(`Sending ${photo ? 'photo+text' : 'text only'} to Speakaboo for user ${userId}`);
      const response = await fetch(SPEAKABOO_API_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      const text = await response.text().catch(() => '');
      this.logger.info(`Speakaboo API responded in ${Date.now() - startTime}ms`);
      if (!response.ok) {
        // If Speakaboo requires an image and we sent text-only, capture a fresh photo and retry once
        if (response.status === 412 && !photo) {
          this.logger.warn(`Speakaboo 412 (needs image). Capturing fresh photo for user ${userId} and retrying once.`);
          try {
            // If camera already capturing, let cachePhoto pick up the queued question
            if (this.cameraRequestInFlight.get(userId)) {
              this.logger.info(`Camera in-flight for ${userId}; queuing prompt to send with next photo.`);
              this.pendingQuestionTextByUserId.set(userId, promptToSend);
              return;
            }

            const session = this.sessionsByUserId.get(userId);
            if (!session) {
              this.logger.error(`No active session for user ${userId}; cannot capture photo to retry.`);
              return;
            }

            this.cameraRequestInFlight.set(userId, true);
            const newPhoto = await session.camera.requestPhoto();
            const retryStoredPhoto: StoredPhoto = {
              requestId: newPhoto.requestId,
              buffer: newPhoto.buffer,
              timestamp: newPhoto.timestamp,
              userId: userId,
              mimeType: newPhoto.mimeType,
              filename: newPhoto.filename,
              size: newPhoto.size,
            };

            // Update latest cached photo without triggering forward handler
            this.photos.set(userId, retryStoredPhoto);
            this.latestPhotoTimestamp.set(userId, retryStoredPhoto.timestamp.getTime());

            const retryPayload: any = {
              text: promptToSend,
              image: retryStoredPhoto.buffer.toString('base64'),
              filename: retryStoredPhoto.filename,
              mimeType: retryStoredPhoto.mimeType,
              timestamp: retryStoredPhoto.timestamp.toISOString(),
            };

            this.logger.info(`Retrying Speakaboo with fresh photo for user ${userId}`);
            const retryResponse = await fetch(SPEAKABOO_API_URL, {
              method: 'POST',
              headers,
              body: JSON.stringify(retryPayload),
            });

            const retryText = await retryResponse.text().catch(() => '');
            if (!retryResponse.ok) {
              this.logger.error(`Retry Speakaboo failed with ${retryResponse.status}: ${retryText}`);
              this.stopLoadingSfx(userId);
              await this.stopAllAudio(userId);
              return;
            }

            let retryAnswer = '';
            try {
              const parsed = JSON.parse(retryText);
              if (typeof parsed === 'object' && parsed) {
                if (typeof (parsed as any).answer === 'string') {
                  retryAnswer = (parsed as any).answer;
                } else if ((parsed as any).data && typeof (parsed as any).data.answer === 'string') {
                  retryAnswer = (parsed as any).data.answer;
                } else if (typeof parsed === 'string') {
                  retryAnswer = parsed as unknown as string;
                }
              }
            } catch (_) {
              retryAnswer = retryText || '';
            }

            if (retryAnswer) {
              this.latestAnalysis.set(userId, { requestId: retryStoredPhoto.requestId, answer: retryAnswer, raw: retryText });

              if (SPEAK_ANALYSIS) {
                const sessionForRetry = this.sessionsByUserId.get(userId);
                if (sessionForRetry) {
                  try {
                    await new Promise((r) => setTimeout(r, 250));
                    this.stopLoadingSfx(userId);
                    await this.stopAllAudio(userId);
                    this.detachTranscription(userId);
                    this.isSpeaking.set(userId, true);
                    this.logger.info(`Starting TTS for user ${userId}: ${Math.min(retryAnswer.length, 120)} chars`);
                    let speakResult: any = await sessionForRetry.audio.speak(retryAnswer, {
                      voice_settings: {
                        speed: 1.2
                      }
                    });
                    if (speakResult && speakResult.success === false) {
                      this.logger.warn(`TTS first attempt failed; retrying once after short backoff`);
                      await new Promise((r) => setTimeout(r, 350));
                      speakResult = await sessionForRetry.audio.speak(retryAnswer, {
                        voice_settings: {
                          speed: 1.2
                        }
                      });
                   }
                    if (speakResult && typeof speakResult.success === 'boolean') {
                      if (speakResult.success) {
                        this.logger.info(`Finished TTS for user ${userId}${speakResult.duration ? `, duration ${speakResult.duration}ms` : ''}`);
                      } else {
                        this.logger.error(`TTS reported failure for user ${userId}${speakResult.error ? `: ${speakResult.error}` : ''}`);
                      }
                    } else {
                      this.logger.info(`Finished TTS for user ${userId}`);
                    }
                  } 
                  catch (speakErr) {
                    this.logger.error(`Failed to speak analysis: ${speakErr}`);
                  } finally {
                    this.isSpeaking.set(userId, false);
                    this.reattachTranscriptionIfNeeded(userId);
                  }
                }
              }
            }

            this.logger.info(`Forwarded retry photo ${retryStoredPhoto.requestId} to Speakaboo successfully.`);
            return; // Completed via retry path
          } catch (retryErr) {
            this.logger.error(`Retry with fresh photo failed: ${retryErr}`);
            this.stopLoadingSfx(userId);
            await this.stopAllAudio(userId);
            return;
          } finally {
            this.cameraRequestInFlight.set(userId, false);
          }
        }

        this.logger.error(`Speakaboo API responded with ${response.status}: ${text}`);
        this.stopLoadingSfx(userId);
        await this.stopAllAudio(userId);
        return;
      }

      let answer = '';
      try {
        const parsed = JSON.parse(text);
        if (typeof parsed === 'object' && parsed) {
          if (typeof (parsed as any).answer === 'string') {
            answer = (parsed as any).answer;
          } else if ((parsed as any).data && typeof (parsed as any).data.answer === 'string') {
            answer = (parsed as any).data.answer;
          } else if (typeof parsed === 'string') {
            answer = parsed as unknown as string;
          }
        }
      } catch (_) {
        answer = text || '';
      }

      if (answer) {
        this.latestAnalysis.set(userId, { requestId: photo ? photo.requestId : 'transcription', answer, raw: text });

        if (SPEAK_ANALYSIS) {
          const session = this.sessionsByUserId.get(userId);
          if (session) {
            try {
              await new Promise((r) => setTimeout(r, 250));
              this.stopLoadingSfx(userId);
              await this.stopAllAudio(userId);
              this.detachTranscription(userId);
              this.isSpeaking.set(userId, true);
              this.logger.info(`Starting TTS for user ${userId}: ${Math.min(answer.length, 120)} chars`);

              let speakResult: any = await session.audio.speak(answer, {
                voice_settings: {
                  speed: 1.2
                }
              });
              if (speakResult && speakResult.success === false) {
                this.logger.warn(`TTS first attempt failed; retrying once after short backoff`);
                await new Promise((r) => setTimeout(r, 350));
                speakResult = await session.audio.speak(answer, {
                  voice_settings: {
                    speed: 1.2
                  }
                });
              }
              if (speakResult && typeof speakResult.success === 'boolean') {
                if (speakResult.success) {
                  this.logger.info(`Finished TTS for user ${userId}${speakResult.duration ? `, duration ${speakResult.duration}ms` : ''}`);
                } else {
                  this.logger.error(`TTS reported failure for user ${userId}${speakResult.error ? `: ${speakResult.error}` : ''}`);
                }
              } else {
                this.logger.info(`Finished TTS for user ${userId}`);
              }
           } 
            catch (speakErr) {
              this.logger.error(`Failed to speak analysis: ${speakErr}`);
            } finally {
              this.isSpeaking.set(userId, false);
              this.reattachTranscriptionIfNeeded(userId);
            }
          }
        }
      }

      this.logger.info(`Forwarded ${photo ? `photo ${photo.requestId}` : 'transcription'} to Speakaboo successfully.`);
    } catch (error) {
      this.logger.error(`Failed forwarding to Speakaboo: ${error}`);
      this.stopLoadingSfx(userId);
      await this.stopAllAudio(userId);
    }
    finally {
      this.isSendingToSpeakaboo.set(userId, false);
    }
  }


  /**
 * Set up webview routes for photo display functionality
 */
  private setupWebviewRoutes(): void {
    const app = this.getExpressApp();

    // Expose static files (e.g., loading sfx) if PUBLIC_BASE_URL points back to this server
    try {
      const expressStatic = require('express').static;
      app.use('/static', expressStatic(path.join(process.cwd(), 'sounds')));
    } catch (_) {}


    // API endpoint to get the latest photo for the authenticated user
    app.get('/api/latest-photo', (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;

      if (!userId) {
        res.status(401).json({ error: 'Not authenticated' });
        return;
      }

      const photo = this.photos.get(userId);
      if (!photo) {
        res.status(404).json({ error: 'No photo available' });
        return;
      }

      res.json({
        requestId: photo.requestId,
        timestamp: photo.timestamp.getTime(),
        hasPhoto: true
      });
    });

    // API endpoint to get photo data
    app.get('/api/photo/:requestId', (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;
      const requestId = req.params.requestId;

      if (!userId) {
        res.status(401).json({ error: 'Not authenticated' });
        return;
      }

      const photo = this.photos.get(userId);
      if (!photo || photo.requestId !== requestId) {
        res.status(404).json({ error: 'Photo not found' });
        return;
      }

      res.set({
        'Content-Type': photo.mimeType,
        'Cache-Control': 'no-cache'
      });
      res.send(photo.buffer);
    });

    // API endpoint to get latest analysis
    app.get('/api/analysis', (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;
      if (!userId) {
        res.status(401).json({ error: 'Not authenticated' });
        return;
      }
      const analysis = this.latestAnalysis.get(userId);
      if (!analysis) {
        res.status(404).json({ error: 'No analysis available' });
        return;
      }
      res.json(analysis);
    });

    // Main webview route - displays the photo viewer interface
    app.get('/webview', async (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;

      if (!userId) {
        res.status(401).send(`
          <html>
            <head><title>Photo Viewer - Not Authenticated</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
              <h1>Please open this page from the MentraOS app</h1>
            </body>
          </html>
        `);
        return;
      }

      const templatePath = path.join(process.cwd(), 'views', 'photo-viewer.ejs');
      const html = await ejs.renderFile(templatePath, {});
      res.send(html);
    });

    // Dev-only unauthenticated endpoints to enable desktop debugging
    if (ALLOW_DEV_VIEW) {
      this.logger.warn('ALLOW_DEV_VIEW enabled: dev endpoints are accessible without Mentra auth');

      // Unauthenticated webview that reads userId from query string
      app.get('/dev/webview', async (req: any, res: any) => {
        const templatePath = path.join(process.cwd(), 'views', 'photo-viewer.ejs');
        const html = await ejs.renderFile(templatePath, {});
        res.send(html);
      });

      // Unauthenticated latest-photo endpoint using userId query param
      app.get('/dev/latest-photo', (req: any, res: any) => {
        const userId = (req.query.userId as string) || '';
        if (!userId) {
          res.status(400).json({ error: 'Missing userId' });
          return;
        }

        const photo = this.photos.get(userId);
        if (!photo) {
          res.status(404).json({ error: 'No photo available' });
          return;
        }

        res.json({
          requestId: photo.requestId,
          timestamp: photo.timestamp.getTime(),
          hasPhoto: true
        });
      });

      // Unauthenticated photo fetch using userId query param
      app.get('/dev/photo/:requestId', (req: any, res: any) => {
        const userId = (req.query.userId as string) || '';
        const requestId = req.params.requestId;

        if (!userId) {
          res.status(400).json({ error: 'Missing userId' });
          return;
        }

        const photo = this.photos.get(userId);
        if (!photo || photo.requestId !== requestId) {
          res.status(404).json({ error: 'Photo not found' });
          return;
        }

        res.set({
          'Content-Type': photo.mimeType,
          'Cache-Control': 'no-cache'
        });
        res.send(photo.buffer);
      });

      // Unauthenticated analysis fetch using userId query param
      app.get('/dev/analysis', (req: any, res: any) => {
        const userId = (req.query.userId as string) || '';
        if (!userId) {
          res.status(400).json({ error: 'Missing userId' });
          return;
        }
        const analysis = this.latestAnalysis.get(userId);
        if (!analysis) {
          res.status(404).json({ error: 'No analysis available' });
          return;
        }
        res.json(analysis);
      });
    }
  }
}



// Start the server
// DEV CONSOLE URL: https://console.mentra.glass/
// Get your webhook URL from ngrok (or whatever public URL you have)
const app = new ExampleMentraOSApp();

app.start().catch(console.error);