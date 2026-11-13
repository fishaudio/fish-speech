// Speaker audio + transcripts, kept in this browser. IndexedDB holds the raw
// bytes (localStorage's quota is too small for audio); JSON import/export is
// the only way to move a voice to another browser or machine.

const DB_NAME = 'fish-speech.voice-cache'
const STORE = 'state'
const WORKSPACE_KEY = 'workspace'
const VOICES_KEY = 'voices'
const FILE_VERSION = 1

const MAX_VOICES = 64
const MAX_CLIPS = 16
const MAX_AUDIO_CHARS = 32 * 1024 * 1024

export type StoredClip = {
  name: string
  mime: string
  text: string
  audio: ArrayBuffer
}

export type StoredSpeaker = {
  name: string
  clips: StoredClip[]
}

export type SavedVoice = StoredSpeaker & {
  id: string
  savedAt: number
  source?: 'server' | 'browser'
  serverId?: string
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

function base64ToArrayBuffer(value: string): ArrayBuffer {
  const binary = atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes.buffer
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE)
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function withStore<T>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb()
  try {
    const tx = db.transaction(STORE, mode)
    const request = run(tx.objectStore(STORE))
    const result = await new Promise<T>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
      tx.onabort = () => reject(tx.error)
    })
    return result
  } finally {
    db.close()
  }
}

function isStoredSpeaker(value: unknown): value is StoredSpeaker {
  if (!value || typeof value !== 'object') {
    return false
  }
  const speaker = value as Partial<StoredSpeaker>
  return (
    typeof speaker.name === 'string' &&
    Array.isArray(speaker.clips) &&
    speaker.clips.every((clip) => {
      if (!clip || typeof clip !== 'object') {
        return false
      }
      const item = clip as Partial<StoredClip>
      return (
        typeof item.name === 'string' &&
        typeof item.mime === 'string' &&
        typeof item.text === 'string' &&
        item.audio instanceof ArrayBuffer
      )
    })
  )
}

// The current Voices panel is written on every change so reopening the page
// puts the user back where they left off.
export async function readWorkspace(): Promise<StoredSpeaker[]> {
  const value: unknown = await withStore('readonly', (store) => store.get(WORKSPACE_KEY))
  if (!Array.isArray(value) || !value.every(isStoredSpeaker)) {
    return []
  }
  return value
}

export async function writeWorkspace(speakers: StoredSpeaker[]): Promise<void> {
  await withStore('readwrite', (store) => store.put(speakers, WORKSPACE_KEY))
}

export async function readVoices(): Promise<SavedVoice[]> {
  const value: unknown = await withStore('readonly', (store) => store.get(VOICES_KEY))
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(
    (voice): voice is SavedVoice =>
      isStoredSpeaker(voice) &&
      typeof (voice as SavedVoice).id === 'string' &&
      typeof (voice as SavedVoice).savedAt === 'number',
  )
}

export async function writeVoices(voices: SavedVoice[]): Promise<void> {
  await withStore('readwrite', (store) => store.put(voices, VOICES_KEY))
}

// Voices captured by the local server from /v1/tts requests live in the
// repo's .app_cache directory. They are shared by everyone using that server.
type ServerEntry = {
  id: string
  name: string
  text: string
  mime: string
  created_at: string
  audio: string
}

export async function fetchServerVoiceCache(endpoint: string): Promise<SavedVoice[]> {
  const response = await fetch(`${endpoint}/v1/speaker-cache`, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`Server cache unavailable (${response.status})`)
  }
  const data: unknown = await response.json()
  if (!data || typeof data !== 'object' || !Array.isArray((data as { entries?: unknown }).entries)) {
    throw new Error('Invalid speaker cache response.')
  }

  return (data as { entries: unknown[] }).entries.flatMap((entry): SavedVoice[] => {
    if (!entry || typeof entry !== 'object') {
      return []
    }
    const item = entry as Partial<ServerEntry>
    if (
      typeof item.id !== 'string' ||
      typeof item.audio !== 'string' ||
      typeof item.text !== 'string'
    ) {
      return []
    }
    let audio: ArrayBuffer
    try {
      audio = base64ToArrayBuffer(item.audio)
    } catch {
      return []
    }
    const name = typeof item.name === 'string' && item.name ? item.name : 'Reference clip'
    const mime =
      typeof item.mime === 'string' && item.mime.startsWith('audio/')
        ? item.mime
        : 'audio/mpeg'
    return [
      {
        id: `server-${item.id}`,
        serverId: item.id,
        source: 'server',
        savedAt: Date.parse(item.created_at ?? '') || Date.now(),
        name,
        clips: [{ name, mime, text: item.text, audio }],
      },
    ]
  })
}

export async function deleteServerVoice(endpoint: string, serverId: string): Promise<void> {
  const response = await fetch(
    `${endpoint}/v1/speaker-cache/${encodeURIComponent(serverId)}`,
    { method: 'DELETE' },
  )
  if (!response.ok) {
    throw new Error(`Could not delete cache entry (${response.status})`)
  }
}

export function exportVoices(voices: SavedVoice[]): string {
  return JSON.stringify(
    {
      version: FILE_VERSION,
      exportedAt: new Date().toISOString(),
      voices: voices.map((voice) => ({
        name: voice.name,
        clips: voice.clips.map((clip) => ({
          name: clip.name,
          mime: clip.mime,
          text: clip.text,
          audio: arrayBufferToBase64(clip.audio),
        })),
      })),
    },
    null,
    2,
  )
}

// Imported files are untrusted; reject anything that isn't a well-formed,
// reasonably sized voice before it reaches IndexedDB.
export function parseVoiceFile(raw: string): SavedVoice[] {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error('Not a valid JSON file.')
  }

  if (
    !parsed ||
    typeof parsed !== 'object' ||
    (parsed as { version?: unknown }).version !== FILE_VERSION ||
    !Array.isArray((parsed as { voices?: unknown }).voices)
  ) {
    throw new Error('Unrecognised voice file.')
  }

  const entries = (parsed as { voices: unknown[] }).voices
  if (entries.length === 0 || entries.length > MAX_VOICES) {
    throw new Error('The file contains no voices, or too many.')
  }

  const now = Date.now()
  return entries.map((entry, index) => {
    if (
      !entry ||
      typeof entry !== 'object' ||
      typeof (entry as { name?: unknown }).name !== 'string' ||
      !Array.isArray((entry as { clips?: unknown }).clips)
    ) {
      throw new Error(`Voice ${index + 1} is malformed.`)
    }
    const name = (entry as { name: string }).name.trim()
    const clips = (entry as { clips: unknown[] }).clips
    if (clips.length === 0 || clips.length > MAX_CLIPS) {
      throw new Error(`Voice "${name}" has an invalid number of clips.`)
    }

    return {
      id: `import-${now}-${index}`,
      savedAt: now,
      name: name || `Voice ${index + 1}`,
      clips: clips.map((clip, clipIndex) => {
        if (
          !clip ||
          typeof clip !== 'object' ||
          typeof (clip as { name?: unknown }).name !== 'string' ||
          typeof (clip as { mime?: unknown }).mime !== 'string' ||
          typeof (clip as { text?: unknown }).text !== 'string' ||
          typeof (clip as { audio?: unknown }).audio !== 'string'
        ) {
          throw new Error(`Clip ${clipIndex + 1} in "${name}" is malformed.`)
        }
        const item = clip as { name: string; mime: string; text: string; audio: string }
        if (item.audio.length === 0 || item.audio.length > MAX_AUDIO_CHARS) {
          throw new Error(`Clip "${item.name}" is empty or too large.`)
        }
        let audio: ArrayBuffer
        try {
          audio = base64ToArrayBuffer(item.audio)
        } catch {
          throw new Error(`Clip "${item.name}" has invalid audio data.`)
        }
        return {
          name: item.name,
          mime: item.mime || 'audio/mpeg',
          text: item.text,
          audio,
        }
      }),
    }
  })
}
