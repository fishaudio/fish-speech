import { useEffect, useRef, useState } from 'react'
import { Download, LoaderCircle, Moon, Pause, Pencil, Play, Plus, Sun, Trash2, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { cn, readTheme, saveTheme, type Theme } from '@/lib/utils'

type ModelId = 's2-pro' | 'drama-3'
type ModelFamily = 's2' | 'drama3'
type AudioFormat = 'mp3' | 'wav' | 'pcm' | 'opus'
type LatencyMode = 'normal' | 'balanced' | 'low'

type ModelSpec = {
  id: ModelId
  label: string
  // What a local server reports as model_family in GET /health.
  family: ModelFamily
  note: string
  // Drama 3 expects a space after a speaker tag, S2-Pro does not.
  tagSuffix: string
}

const MODELS: ModelSpec[] = [
  {
    id: 's2-pro',
    label: 'S2-Pro',
    family: 's2',
    note: 'Bracketed cues like [whisper] or [laughing] shape the delivery.',
    tagSuffix: '',
  },
  {
    id: 'drama-3',
    label: 'Drama 3',
    family: 'drama3',
    note: 'Made for scripted dialogue. Long scripts are split by the model itself.',
    tagSuffix: ' ',
  },
]

const MODEL_MISMATCH = 'model not match'

function getModel(id: ModelId): ModelSpec {
  return MODELS.find((model) => model.id === id) ?? MODELS[0]
}

function modelForFamily(family: ModelFamily | null): ModelSpec | null {
  return MODELS.find((model) => model.family === family) ?? null
}

const defaultInputText = `[soft] I kept the letter for three years before I opened it. [pause] When I finally did, it said only one thing: "Come home whenever you're ready."`

type OutputSettings = {
  format: AudioFormat
  latency: LatencyMode
  normalize: boolean
}

type Metrics = {
  firstByteMs: number
  bytes: number
  totalMs: number | null
}

type Notice = {
  tone: 'error' | 'info'
  title: string
  detail?: string
}

type Result = {
  url: string
  fileName: string
  playable: boolean
}

type ReferenceItem = {
  id: number
  name: string
  audio: ArrayBuffer
  text: string
  previewUrl: string
}

type SpeakerGroup = {
  id: number
  references: ReferenceItem[]
}

type PendingReference = {
  mode: 'create' | 'edit'
  speakerId: number
  referenceId?: number
  name: string
  mime?: string
  audio?: ArrayBuffer
  text: string
}

// The server accepts <|speaker:0|> to <|speaker:15|>.
const MAX_SPEAKERS = 16

const FISH_AUDIO_ENDPOINT = 'https://api.fish.audio'
const LOCAL_ENDPOINT_HINT = 'http://127.0.0.1:8080'
const SETTINGS_STORAGE_KEY = 'fish-speech.connection.v1'

type ConnectionSettings = {
  endpoint: string
  apiKey: string
  model: ModelId
}

type LocalProbe =
  | { status: 'offline' }
  | { status: 'loading' }
  | { status: 'ready'; family: ModelFamily | null }

type ServerState = LocalProbe | { status: 'checking' }

function normalizeEndpoint(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '').replace(/\/v1\/tts$/, '')
  return trimmed || FISH_AUDIO_ENDPOINT
}

function isLocalEndpoint(endpoint: string): boolean {
  return /\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?(?:\/|$)/i.test(endpoint)
}

function isModelId(value: unknown): value is ModelId {
  return MODELS.some((model) => model.id === value)
}

function readConnectionSettings(): ConnectionSettings {
  const fallback: ConnectionSettings = {
    endpoint: FISH_AUDIO_ENDPOINT,
    apiKey: '',
    model: MODELS[0].id,
  }
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY)
    if (!raw) {
      return fallback
    }
    const parsed = JSON.parse(raw) as Partial<ConnectionSettings>
    return {
      endpoint: normalizeEndpoint(
        typeof parsed.endpoint === 'string' ? parsed.endpoint : FISH_AUDIO_ENDPOINT,
      ),
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : '',
      model: isModelId(parsed.model) ? parsed.model : fallback.model,
    }
  } catch {
    return fallback
  }
}

function writeConnectionSettings(settings: ConnectionSettings): void {
  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
  } catch {
    // Storage can be disabled (private mode); the in-memory setting still works.
  }
}

async function probeServer(endpoint: string): Promise<LocalProbe> {
  try {
    const response = await fetch(`${endpoint}/health`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    })
    if (!response.ok) {
      return { status: 'offline' }
    }
    const body = (await response.json()) as { status?: string; model_family?: unknown }
    if (body.status !== 'ok') {
      return { status: 'loading' }
    }
    const family =
      body.model_family === 's2' || body.model_family === 'drama3' ? body.model_family : null
    return { status: 'ready', family }
  } catch {
    return { status: 'offline' }
  }
}

class NoticeError extends Error {
  title: string

  constructor(title: string, detail: string) {
    super(detail)
    this.title = title
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const data: unknown = await response.json()
    if (data && typeof data === 'object') {
      const { detail, message } = data as { detail?: unknown; message?: unknown }
      if (typeof detail === 'string') {
        return detail
      }
      if (Array.isArray(detail)) {
        return detail
          .map((item) =>
            item && typeof item === 'object' && 'msg' in item ? String(item.msg) : String(item),
          )
          .join('; ')
      }
      if (typeof message === 'string') {
        return message
      }
    }
  } catch {
    // Not JSON.
  }
  return response.statusText || 'The server returned an error.'
}

const formatMimeMap: Record<AudioFormat, string> = {
  mp3: 'audio/mpeg',
  wav: 'audio/wav',
  pcm: 'audio/pcm',
  opus: 'audio/opus',
}

function createId() {
  return Date.now() + Math.floor(Math.random() * 100000)
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

function createSpeakerGroup(): SpeakerGroup {
  return { id: createId(), references: [] }
}

function buildReferencesPayload(speakerGroups: SpeakerGroup[], includeBinaryAudio: boolean) {
  // Speaker N is the Nth group; the server tags transcripts that have no tag.
  return speakerGroups.flatMap((speakerGroup, speaker) =>
    speakerGroup.references.map((reference) => ({
      text: reference.text,
      speaker,
      audio: includeBinaryAudio ? arrayBufferToBase64(reference.audio) : '<audio binary data>',
    })),
  )
}

// Local servers only read text and references; the rest is hosted-API only.
function buildRequestBody(
  text: string,
  local: boolean,
  output: OutputSettings,
  speakerGroups: SpeakerGroup[],
  includeBinaryAudio: boolean,
) {
  const references = buildReferencesPayload(speakerGroups, includeBinaryAudio)
  if (local) {
    return { text, references }
  }
  return {
    text,
    references,
    format: output.format,
    latency: output.latency,
    normalize: output.normalize,
  }
}

function buildRequestHeaders(connection: ConnectionSettings, redactKey: boolean) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    model: connection.model,
  }
  if (connection.apiKey) {
    headers.Authorization = `Bearer ${redactKey ? '<api key>' : connection.apiKey}`
  }
  return headers
}

function createFileName(inputText: string) {
  return inputText.trim().replace(/\s+/g, '-').slice(0, 24) || 'tts'
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Unknown error'
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatMs(ms: number) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function hostOf(endpoint: string) {
  try {
    return new URL(endpoint).host
  } catch {
    return endpoint
  }
}

function waitForSourceBuffer(sourceBuffer: SourceBuffer) {
  if (!sourceBuffer.updating) {
    return Promise.resolve()
  }
  return new Promise<void>((resolve) => {
    sourceBuffer.addEventListener('updateend', () => resolve(), { once: true })
  })
}

function canUseStreamingPlayback(format: AudioFormat) {
  const mime = formatMimeMap[format]
  return typeof window.MediaSource !== 'undefined' && MediaSource.isTypeSupported(mime)
}

const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.userAgent)

type SegmentedProps<T extends string> = {
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
  itemClassName?: string
  ariaLabel: string
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
  itemClassName,
  ariaLabel,
}: SegmentedProps<T>) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      aria-label={ariaLabel}
      onValueChange={(next) => {
        if (next) {
          onChange(next as T)
        }
      }}
      className="inline-flex gap-0.5 rounded-lg bg-muted p-0.5"
    >
      {options.map((option) => (
        <ToggleGroupItem
          key={option.value}
          value={option.value}
          className={cn(
            'h-7 rounded-md border-0 bg-transparent px-3 text-xs text-muted-foreground hover:bg-transparent hover:text-foreground data-[state=on]:bg-card data-[state=on]:text-foreground data-[state=on]:shadow-sm',
            itemClassName,
          )}
        >
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  )
}

function IconButton({
  label,
  className,
  ...props
}: React.ComponentProps<'button'> & { label: string }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      className={cn(
        'flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-40',
        className,
      )}
      {...props}
    />
  )
}

const initialSpeakerGroup = createSpeakerGroup()

function App() {
  const [inputText, setInputText] = useState(defaultInputText)
  const [output, setOutput] = useState<OutputSettings>({
    format: 'mp3',
    latency: 'normal',
    normalize: true,
  })
  const [speakerGroups, setSpeakerGroups] = useState<SpeakerGroup[]>([initialSpeakerGroup])
  const [pendingReference, setPendingReference] = useState<PendingReference | null>(null)
  const [playingReferenceId, setPlayingReferenceId] = useState<number | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isStreamingAudio, setIsStreamingAudio] = useState(false)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [isRequestOpen, setIsRequestOpen] = useState(false)
  const [copyLabel, setCopyLabel] = useState('Copy')
  const [connection, setConnection] = useState<ConnectionSettings>(readConnectionSettings)
  const [serverState, setServerState] = useState<ServerState>({ status: 'checking' })
  const [probeNonce, setProbeNonce] = useState(0)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [endpointDraft, setEndpointDraft] = useState(connection.endpoint)
  const [apiKeyDraft, setApiKeyDraft] = useState(connection.apiKey)
  const [theme, setTheme] = useState<Theme>(readTheme)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const previewPlayerRef = useRef<HTMLAudioElement | null>(null)
  const speakerGroupsRef = useRef<SpeakerGroup[]>([])
  const uploadTargetSpeakerIdRef = useRef<number | null>(null)
  const resultUrlRef = useRef<string | null>(null)
  const mediaSourceUrlRef = useRef<string | null>(null)

  speakerGroupsRef.current = speakerGroups

  const usingLocalServer = isLocalEndpoint(connection.endpoint)
  const selectedModel = getModel(connection.model)
  const serverModel =
    serverState.status === 'ready' ? modelForFamily(serverState.family) : null
  const modelMismatch =
    usingLocalServer && serverState.status === 'ready' && serverModel?.id !== selectedModel.id
  const effectiveFormat: AudioFormat = usingLocalServer ? 'wav' : output.format
  const hasMultipleSpeakers = speakerGroups.length > 1

  useEffect(() => {
    if (!isLocalEndpoint(connection.endpoint)) {
      return
    }
    let cancelled = false
    void probeServer(connection.endpoint).then((state) => {
      if (!cancelled) {
        setServerState(state)
      }
    })
    return () => {
      cancelled = true
    }
  }, [connection.endpoint, probeNonce])

  useEffect(() => {
    return () => {
      previewPlayerRef.current?.pause()
      speakerGroupsRef.current.forEach((speakerGroup) => {
        speakerGroup.references.forEach((reference) => {
          URL.revokeObjectURL(reference.previewUrl)
        })
      })
      if (resultUrlRef.current) {
        URL.revokeObjectURL(resultUrlRef.current)
      }
      if (mediaSourceUrlRef.current) {
        URL.revokeObjectURL(mediaSourceUrlRef.current)
      }
    }
  }, [])

  function toggleTheme() {
    setTheme((current) => {
      const next = current === 'dark' ? 'light' : 'dark'
      saveTheme(next)
      return next
    })
  }

  function updateConnection(next: ConnectionSettings) {
    writeConnectionSettings(next)
    setConnection(next)
  }

  function selectModel(model: ModelId) {
    updateConnection({ ...connection, model })
    if (notice?.title === MODEL_MISMATCH) {
      setNotice(null)
    }
  }

  function openSettings() {
    setEndpointDraft(connection.endpoint)
    setApiKeyDraft(connection.apiKey)
    setIsSettingsOpen(true)
  }

  function saveSettings() {
    const next: ConnectionSettings = {
      ...connection,
      endpoint: normalizeEndpoint(endpointDraft),
      apiKey: apiKeyDraft.trim(),
    }
    updateConnection(next)
    if (isLocalEndpoint(next.endpoint)) {
      setServerState({ status: 'checking' })
      setProbeNonce((current) => current + 1)
    }
    setNotice(null)
    setIsSettingsOpen(false)
  }

  function insertSpeakerTag(speakerIndex: number) {
    const tag = `<|speaker:${speakerIndex}|>${selectedModel.tagSuffix}`
    const textarea = textareaRef.current
    const start = textarea?.selectionStart ?? inputText.length
    const end = textarea?.selectionEnd ?? inputText.length
    setInputText(inputText.slice(0, start) + tag + inputText.slice(end))
    window.requestAnimationFrame(() => {
      if (textarea) {
        const caret = start + tag.length
        textarea.focus()
        textarea.setSelectionRange(caret, caret)
      }
    })
  }

  function addSpeaker() {
    setSpeakerGroups((current) => [...current, createSpeakerGroup()])
  }

  function stopPreview() {
    previewPlayerRef.current?.pause()
    setPlayingReferenceId(null)
  }

  function removeSpeaker(speakerId: number) {
    const target = speakerGroups.find((speakerGroup) => speakerGroup.id === speakerId)
    if (target?.references.some((reference) => reference.id === playingReferenceId)) {
      stopPreview()
    }
    target?.references.forEach((reference) => URL.revokeObjectURL(reference.previewUrl))

    setSpeakerGroups((current) => {
      const next = current.filter((speakerGroup) => speakerGroup.id !== speakerId)
      return next.length > 0 ? next : [createSpeakerGroup()]
    })
    if (pendingReference?.speakerId === speakerId) {
      setPendingReference(null)
    }
  }

  function addReference(
    speakerId: number,
    name: string,
    audio: ArrayBuffer,
    mime: string,
    text: string,
  ) {
    const previewUrl = URL.createObjectURL(new Blob([audio], { type: mime }))
    setSpeakerGroups((current) =>
      current.map((speakerGroup) =>
        speakerGroup.id === speakerId
          ? {
              ...speakerGroup,
              references: [
                ...speakerGroup.references,
                { id: createId(), name, audio, text, previewUrl },
              ],
            }
          : speakerGroup,
      ),
    )
  }

  function removeReference(speakerId: number, reference: ReferenceItem) {
    if (reference.id === playingReferenceId) {
      stopPreview()
    }
    URL.revokeObjectURL(reference.previewUrl)
    setSpeakerGroups((current) =>
      current.map((speakerGroup) =>
        speakerGroup.id === speakerId
          ? {
              ...speakerGroup,
              references: speakerGroup.references.filter((item) => item.id !== reference.id),
            }
          : speakerGroup,
      ),
    )
  }

  function updateReferenceText(speakerId: number, referenceId: number, text: string) {
    setSpeakerGroups((current) =>
      current.map((speakerGroup) =>
        speakerGroup.id === speakerId
          ? {
              ...speakerGroup,
              references: speakerGroup.references.map((reference) =>
                reference.id === referenceId ? { ...reference, text } : reference,
              ),
            }
          : speakerGroup,
      ),
    )
  }

  function togglePreview(reference: ReferenceItem) {
    let player = previewPlayerRef.current
    if (!player) {
      player = new Audio()
      player.addEventListener('ended', () => setPlayingReferenceId(null))
      previewPlayerRef.current = player
    }
    if (playingReferenceId === reference.id) {
      stopPreview()
      return
    }
    player.pause()
    player.src = reference.previewUrl
    setPlayingReferenceId(reference.id)
    void player.play().catch(() => setPlayingReferenceId(null))
  }

  function pickClip(speakerId: number) {
    uploadTargetSpeakerIdRef.current = speakerId
    fileInputRef.current?.click()
  }

  async function handleReferenceUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    const speakerId = uploadTargetSpeakerIdRef.current
    event.target.value = ''
    uploadTargetSpeakerIdRef.current = null

    if (!file || typeof speakerId !== 'number') {
      return
    }

    const audio = await file.arrayBuffer()
    setPendingReference({
      mode: 'create',
      speakerId,
      name: file.name,
      mime: file.type || 'audio/mpeg',
      audio,
      text: '',
    })
  }

  function savePendingReference() {
    if (!pendingReference) {
      return
    }
    if (pendingReference.mode === 'create' && pendingReference.audio) {
      addReference(
        pendingReference.speakerId,
        pendingReference.name,
        pendingReference.audio,
        pendingReference.mime ?? 'audio/mpeg',
        pendingReference.text.trim(),
      )
    }
    if (pendingReference.mode === 'edit' && typeof pendingReference.referenceId === 'number') {
      updateReferenceText(
        pendingReference.speakerId,
        pendingReference.referenceId,
        pendingReference.text.trim(),
      )
    }
    setPendingReference(null)
  }

  function clearResult() {
    if (resultUrlRef.current) {
      URL.revokeObjectURL(resultUrlRef.current)
      resultUrlRef.current = null
    }
    setResult(null)
  }

  function clearMediaSourceUrl() {
    if (mediaSourceUrlRef.current) {
      URL.revokeObjectURL(mediaSourceUrlRef.current)
      mediaSourceUrlRef.current = null
    }
  }

  function buildRequestPreview() {
    return JSON.stringify(
      {
        method: 'POST',
        url: `${connection.endpoint}/v1/tts`,
        headers: buildRequestHeaders(connection, true),
        body: buildRequestBody(inputText.trim(), usingLocalServer, output, speakerGroups, false),
      },
      null,
      2,
    )
  }

  async function copyRequestPreview() {
    try {
      await navigator.clipboard.writeText(buildRequestPreview())
      setCopyLabel('Copied')
    } catch {
      setCopyLabel('Copy failed')
    }
    window.setTimeout(() => setCopyLabel('Copy'), 2000)
  }

  // A local server runs exactly one model. Refuse to synthesize unless it is
  // the one selected, so the output never comes from a different model.
  async function assertServerModel() {
    const state = await probeServer(connection.endpoint)
    setServerState(state)
    if (state.status === 'offline') {
      throw new NoticeError(
        'Server unreachable',
        `Nothing answered at ${connection.endpoint}/health.`,
      )
    }
    if (state.status === 'loading') {
      throw new NoticeError('Server not ready', 'The model is still loading. Try again shortly.')
    }
    const running = modelForFamily(state.family)
    if (running?.id !== selectedModel.id) {
      throw new NoticeError(
        MODEL_MISMATCH,
        `${selectedModel.label} is selected, but this server is running ${
          running?.label ?? 'an unrecognised model'
        }.`,
      )
    }
  }

  async function handleGenerate() {
    const audioElement = audioRef.current
    if (!audioElement || isGenerating) {
      return
    }

    const text = inputText.trim()
    if (!text) {
      setNotice({ tone: 'error', title: 'Nothing to read', detail: 'Type or paste some text first.' })
      return
    }
    if (!usingLocalServer && !connection.apiKey) {
      setNotice({
        tone: 'error',
        title: 'API key missing',
        detail: 'Add your Fish Audio API key under Connection.',
      })
      return
    }

    const format = effectiveFormat
    const mime = formatMimeMap[format]
    const useStreamingPlayback = canUseStreamingPlayback(format)

    audioElement.pause()
    clearResult()
    clearMediaSourceUrl()
    setIsStreamingAudio(false)
    setMetrics(null)
    setNotice(null)
    setIsGenerating(true)

    try {
      if (usingLocalServer) {
        await assertServerModel()
      }

      const startTime = performance.now()
      const response = await fetch(`${connection.endpoint}/v1/tts`, {
        method: 'POST',
        headers: buildRequestHeaders(connection, false),
        body: JSON.stringify(
          buildRequestBody(text, usingLocalServer, output, speakerGroups, true),
        ),
      })

      if (!response.ok || !response.body) {
        throw new NoticeError(
          `Request failed (${response.status})`,
          await readErrorDetail(response),
        )
      }

      const reader = response.body.getReader()
      let mediaSource: MediaSource | null = null

      if (useStreamingPlayback) {
        mediaSource = new MediaSource()
        const streamUrl = URL.createObjectURL(mediaSource)
        mediaSourceUrlRef.current = streamUrl
        audioElement.src = streamUrl
        setIsStreamingAudio(true)
      } else {
        audioElement.removeAttribute('src')
        audioElement.load()
      }

      const allChunks: ArrayBuffer[] = []
      const playQueue: ArrayBuffer[] = []
      let readingDone = false
      let receivedLength = 0
      let firstByteMs = -1

      if (mediaSource) {
        const source = mediaSource
        await new Promise<void>((resolve, reject) => {
          source.addEventListener(
            'sourceopen',
            () => {
              try {
                const sourceBuffer = source.addSourceBuffer(mime)

                const processQueue = async () => {
                  while (true) {
                    if (readingDone && playQueue.length === 0) {
                      await waitForSourceBuffer(sourceBuffer)
                      if (source.readyState === 'open') {
                        source.endOfStream()
                      }
                      break
                    }

                    const chunk = playQueue.shift()
                    if (!chunk) {
                      await new Promise<void>((resolveSleep) => {
                        window.setTimeout(resolveSleep, 50)
                      })
                      continue
                    }

                    await waitForSourceBuffer(sourceBuffer)
                    sourceBuffer.appendBuffer(chunk)
                    await waitForSourceBuffer(sourceBuffer)
                  }
                }

                void processQueue()
                resolve()
              } catch (error) {
                reject(error)
              }
            },
            { once: true },
          )
        })
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          readingDone = true
          break
        }

        receivedLength += value.byteLength
        if (firstByteMs < 0) {
          firstByteMs = performance.now() - startTime
        }
        setMetrics({ firstByteMs, bytes: receivedLength, totalMs: null })

        const chunk = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)
        playQueue.push(chunk)
        allChunks.push(chunk)

        if (useStreamingPlayback && audioElement.paused) {
          void audioElement.play().catch(() => undefined)
        }
      }

      const totalMs = performance.now() - startTime
      setMetrics({ firstByteMs, bytes: receivedLength, totalMs })

      const resultUrl = URL.createObjectURL(new Blob(allChunks, { type: mime }))
      resultUrlRef.current = resultUrl
      const playable = format !== 'pcm'
      setResult({ url: resultUrl, fileName: `${createFileName(text)}.${format}`, playable })

      if (!useStreamingPlayback && playable) {
        audioElement.src = resultUrl
        audioElement.load()
        void audioElement.play().catch(() => undefined)
      }
      if (!playable) {
        setNotice({
          tone: 'info',
          title: 'Ready to download',
          detail: 'PCM is raw audio, so the browser cannot play it here.',
        })
      }
    } catch (error) {
      setIsStreamingAudio(false)
      setNotice(
        error instanceof NoticeError
          ? { tone: 'error', title: error.title, detail: error.message }
          : { tone: 'error', title: 'Generation failed', detail: getErrorMessage(error) },
      )
    } finally {
      setIsGenerating(false)
    }
  }

  const connectionStatus: { tone: 'ok' | 'warn' | 'error' | 'idle'; label: string } = (() => {
    if (!usingLocalServer) {
      return connection.apiKey
        ? { tone: 'ok', label: 'Hosted API' }
        : { tone: 'warn', label: 'API key needed' }
    }
    switch (serverState.status) {
      case 'checking':
        return { tone: 'idle', label: 'Checking' }
      case 'offline':
        return { tone: 'error', label: 'Offline' }
      case 'loading':
        return { tone: 'warn', label: 'Loading model' }
      case 'ready':
        return {
          tone: modelMismatch ? 'warn' : 'ok',
          label: `Running ${serverModel?.label ?? 'unknown model'}`,
        }
    }
  })()

  const playerVisible = isStreamingAudio || (result?.playable ?? false)
  const charCount = inputText.trim().length

  return (
    <div className="flex min-h-dvh flex-col lg:h-dvh">
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-4 sm:px-6">
        <div className="flex items-baseline gap-2.5">
          <span className="text-[15px] font-semibold tracking-tight">fish speech</span>
          <span className="hidden text-xs text-muted-foreground sm:inline">text to speech</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={openSettings}
            title="Connection settings"
            className="flex h-8 min-w-0 items-center gap-2 rounded-md border bg-card px-2.5 text-xs transition-colors hover:bg-accent"
          >
            <span
              className={cn(
                'size-1.5 shrink-0 rounded-full',
                connectionStatus.tone === 'ok' && 'bg-emerald-500',
                connectionStatus.tone === 'warn' && 'bg-amber-500',
                connectionStatus.tone === 'error' && 'bg-red-500',
                connectionStatus.tone === 'idle' && 'animate-pulse bg-muted-foreground/50',
              )}
            />
            <span className="max-w-[40vw] truncate font-mono">{hostOf(connection.endpoint)}</span>
            <span className="hidden text-muted-foreground sm:inline">{connectionStatus.label}</span>
          </button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
        </div>
      </header>

      <div className="grid flex-1 lg:min-h-0 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="flex min-w-0 flex-col lg:min-h-0">
          <div className="flex flex-1 flex-col px-4 pb-4 pt-5 sm:px-8 lg:min-h-0">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <Segmented
                ariaLabel="Model"
                value={connection.model}
                options={MODELS.map((model) => ({ value: model.id, label: model.label }))}
                onChange={selectModel}
                itemClassName="h-8 px-4 text-sm"
              />
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {charCount} chars
              </span>
            </div>

            <Textarea
              ref={textareaRef}
              id="inputText"
              aria-label="Text to synthesize"
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault()
                  void handleGenerate()
                }
              }}
              placeholder="Write what should be spoken…"
              spellCheck={false}
              className="min-h-[260px] flex-1 resize-none rounded-lg bg-card p-5 text-[15px] leading-7 shadow-none lg:min-h-0"
            />

            <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
              {modelMismatch ? (
                <p className="text-xs text-amber-700 dark:text-amber-400">
                  This server is running {serverModel?.label ?? 'an unrecognised model'}, so{' '}
                  {selectedModel.label} requests will be refused.
                  {serverModel ? (
                    <>
                      {' '}
                      <button
                        type="button"
                        className="font-medium underline underline-offset-2"
                        onClick={() => selectModel(serverModel.id)}
                      >
                        Use {serverModel.label}
                      </button>
                    </>
                  ) : null}
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">{selectedModel.note}</p>
              )}
              <div className="ml-auto flex items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground"
                  onClick={() => setIsRequestOpen(true)}
                >
                  View request
                </Button>
                <Button
                  type="button"
                  className="min-w-32"
                  onClick={() => void handleGenerate()}
                  disabled={isGenerating}
                >
                  {isGenerating ? <LoaderCircle className="size-4 animate-spin" /> : null}
                  {isGenerating ? 'Generating' : 'Generate'}
                  {!isGenerating ? (
                    <kbd className="ml-1 font-sans text-[11px] opacity-50">
                      {isMac ? '⌘↵' : 'Ctrl ↵'}
                    </kbd>
                  ) : null}
                </Button>
              </div>
            </div>
          </div>

          <div className="border-t bg-card px-4 py-4 sm:px-8">
            <div className="flex min-h-12 flex-col justify-center gap-3">
              {notice ? (
                <div
                  role={notice.tone === 'error' ? 'alert' : 'status'}
                  className={cn(
                    'border-l-2 pl-3',
                    notice.tone === 'error' ? 'border-red-500' : 'border-muted-foreground/40',
                  )}
                >
                  <div
                    className={cn(
                      'text-sm font-medium',
                      notice.tone === 'error' && 'text-red-600 dark:text-red-400',
                    )}
                  >
                    {notice.title}
                  </div>
                  {notice.detail ? (
                    <div className="mt-0.5 text-xs text-muted-foreground">{notice.detail}</div>
                  ) : null}
                </div>
              ) : null}

              <audio
                ref={audioRef}
                controls
                className={cn('h-10 w-full', !playerVisible && 'hidden')}
              />

              {isGenerating && !metrics ? (
                <p className="flex items-center gap-2 text-sm text-muted-foreground">
                  <LoaderCircle className="size-3.5 animate-spin" />
                  {usingLocalServer ? 'Rendering on the local server…' : 'Waiting for audio…'}
                </p>
              ) : null}

              {metrics || result ? (
                <div className="flex flex-wrap items-center justify-between gap-3">
                  {metrics ? (
                    <p className="font-mono text-xs tabular-nums text-muted-foreground">
                      first byte {formatMs(metrics.firstByteMs)} · {formatBytes(metrics.bytes)}
                      {metrics.totalMs !== null ? ` · total ${formatMs(metrics.totalMs)}` : ''}
                    </p>
                  ) : (
                    <span />
                  )}
                  {result ? (
                    <Button asChild variant="outline" size="sm">
                      <a href={result.url} download={result.fileName}>
                        <Download className="size-3.5" />
                        Download {effectiveFormat.toUpperCase()}
                      </a>
                    </Button>
                  ) : null}
                </div>
              ) : null}

              {!notice && !isGenerating && !metrics && !result ? (
                <p className="text-sm text-muted-foreground">
                  Audio will play here once it's generated.
                </p>
              ) : null}
            </div>
          </div>
        </section>

        <aside className="border-t lg:min-h-0 lg:overflow-y-auto lg:border-l lg:border-t-0">
          <div className="px-5 py-5">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Voices</h2>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="-mr-2 text-muted-foreground"
                onClick={addSpeaker}
                disabled={speakerGroups.length >= MAX_SPEAKERS}
              >
                <Plus className="size-3.5" />
                Speaker
              </Button>
            </div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              A short clip and its transcript per speaker. Without one, the model picks a voice.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={handleReferenceUpload}
            />

            <ol className="mt-4 space-y-3">
              {speakerGroups.map((speakerGroup, speakerIndex) => (
                <li key={speakerGroup.id} className="rounded-lg border bg-card">
                  <div className="flex items-center justify-between gap-2 py-1.5 pl-3 pr-1.5">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="text-sm font-medium">Speaker {speakerIndex}</span>
                      {hasMultipleSpeakers ? (
                        <button
                          type="button"
                          title="Insert this tag at the cursor"
                          className="rounded border px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                          onClick={() => insertSpeakerTag(speakerIndex)}
                        >
                          {`<|speaker:${speakerIndex}|>`}
                        </button>
                      ) : null}
                    </div>
                    <div className="flex items-center">
                      <IconButton label="Add clip" onClick={() => pickClip(speakerGroup.id)}>
                        <Plus className="size-4" />
                      </IconButton>
                      {hasMultipleSpeakers ? (
                        <IconButton
                          label={`Remove speaker ${speakerIndex}`}
                          onClick={() => removeSpeaker(speakerGroup.id)}
                        >
                          <X className="size-4" />
                        </IconButton>
                      ) : null}
                    </div>
                  </div>

                  {speakerGroup.references.length > 0 ? (
                    <ul className="divide-y border-t">
                      {speakerGroup.references.map((reference) => (
                        <li key={reference.id} className="flex items-start gap-2.5 py-2.5 pl-3 pr-1.5">
                          <button
                            type="button"
                            aria-label={
                              playingReferenceId === reference.id ? 'Pause clip' : 'Play clip'
                            }
                            className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full border transition-colors hover:bg-accent"
                            onClick={() => togglePreview(reference)}
                          >
                            {playingReferenceId === reference.id ? (
                              <Pause className="size-3" />
                            ) : (
                              <Play className="ml-px size-3" />
                            )}
                          </button>
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[13px] font-medium">{reference.name}</div>
                            {reference.text ? (
                              <p className="line-clamp-2 text-xs leading-5 text-muted-foreground">
                                {reference.text}
                              </p>
                            ) : (
                              <p className="text-xs leading-5 text-amber-700 dark:text-amber-400">
                                No transcript
                              </p>
                            )}
                          </div>
                          <IconButton
                            label="Edit transcript"
                            onClick={() =>
                              setPendingReference({
                                mode: 'edit',
                                speakerId: speakerGroup.id,
                                referenceId: reference.id,
                                name: reference.name,
                                text: reference.text,
                              })
                            }
                          >
                            <Pencil className="size-3.5" />
                          </IconButton>
                          <IconButton
                            label="Remove clip"
                            onClick={() => removeReference(speakerGroup.id, reference)}
                          >
                            <Trash2 className="size-3.5" />
                          </IconButton>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <button
                      type="button"
                      className="block w-full border-t border-dashed px-3 py-3 text-left text-xs text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground"
                      onClick={() => pickClip(speakerGroup.id)}
                    >
                      Add a reference clip
                    </button>
                  )}
                </li>
              ))}
            </ol>
          </div>

          <div className="border-t px-5 py-5">
            <h2 className="text-sm font-semibold">Output</h2>
            {usingLocalServer ? (
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                Local servers return 16-bit mono WAV. Format, latency and text normalization
                apply to the hosted API only.
              </p>
            ) : (
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <Label className="text-xs font-normal text-muted-foreground">Format</Label>
                  <Segmented
                    ariaLabel="Format"
                    value={output.format}
                    options={(['mp3', 'wav', 'pcm', 'opus'] as const).map((format) => ({
                      value: format,
                      label: format,
                    }))}
                    onChange={(format) => setOutput((current) => ({ ...current, format }))}
                    itemClassName="px-2.5 font-mono"
                  />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <Label className="text-xs font-normal text-muted-foreground">Latency</Label>
                  <Segmented
                    ariaLabel="Latency"
                    value={output.latency}
                    options={[
                      { value: 'normal', label: 'Normal' },
                      { value: 'balanced', label: 'Balanced' },
                      { value: 'low', label: 'Low' },
                    ]}
                    onChange={(latency) => setOutput((current) => ({ ...current, latency }))}
                    itemClassName="px-2.5"
                  />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <Label htmlFor="normalize" className="text-xs font-normal text-muted-foreground">
                      Normalize text
                    </Label>
                    <p className="text-[11px] text-muted-foreground/80">
                      Reads numbers, dates and units as words.
                    </p>
                  </div>
                  <Switch
                    id="normalize"
                    checked={output.normalize}
                    onCheckedChange={(normalize) =>
                      setOutput((current) => ({ ...current, normalize }))
                    }
                  />
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>

      <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Connection</DialogTitle>
            <DialogDescription>
              Use Fish Audio's hosted API, or point this page at a server you run yourself.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="endpoint">Endpoint</Label>
              <input
                id="endpoint"
                type="url"
                spellCheck={false}
                value={endpointDraft}
                onChange={(event) => setEndpointDraft(event.target.value)}
                placeholder={FISH_AUDIO_ENDPOINT}
                className="w-full rounded-md border bg-card px-3 py-2 font-mono text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
              />
              <div className="flex flex-wrap gap-1.5">
                {[
                  { label: 'Fish Audio', value: FISH_AUDIO_ENDPOINT },
                  { label: 'Local', value: LOCAL_ENDPOINT_HINT },
                ].map((preset) => (
                  <button
                    key={preset.value}
                    type="button"
                    onClick={() => setEndpointDraft(preset.value)}
                    className={cn(
                      'rounded-md border px-2 py-1 text-xs transition-colors hover:bg-accent',
                      normalizeEndpoint(endpointDraft) === preset.value
                        ? 'border-foreground/30 text-foreground'
                        : 'text-muted-foreground',
                    )}
                  >
                    {preset.label}
                    <span className="ml-1.5 font-mono opacity-60">{hostOf(preset.value)}</span>
                  </button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Base URL only; <code className="font-mono">/v1/tts</code> is added for you.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="apiKey">API key</Label>
              <input
                id="apiKey"
                type="password"
                autoComplete="off"
                value={apiKeyDraft}
                onChange={(event) => setApiKeyDraft(event.target.value)}
                placeholder="Not needed for a local server"
                className="w-full rounded-md border bg-card px-3 py-2 font-mono text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
              />
              <p className="text-xs text-muted-foreground">Kept in this browser's local storage.</p>
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => setIsSettingsOpen(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={saveSettings}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isRequestOpen} onOpenChange={setIsRequestOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Request</DialogTitle>
            <DialogDescription>
              What Generate sends right now. Audio and the API key are left out.
            </DialogDescription>
          </DialogHeader>
          {isRequestOpen ? (
            <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-all rounded-md border bg-muted/50 p-3 font-mono text-xs leading-5">
              {buildRequestPreview()}
            </pre>
          ) : null}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => void copyRequestPreview()}>
              {copyLabel}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={pendingReference !== null}
        onOpenChange={(open) => !open && setPendingReference(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Transcript</DialogTitle>
            <DialogDescription className="truncate">
              {pendingReference
                ? `Speaker ${speakerGroups.findIndex(
                    (speakerGroup) => speakerGroup.id === pendingReference.speakerId,
                  )} · ${pendingReference.name}`
                : ''}
            </DialogDescription>
          </DialogHeader>
          <Textarea
            autoFocus
            value={pendingReference?.text ?? ''}
            onChange={(event) =>
              setPendingReference((current) =>
                current ? { ...current, text: event.target.value } : current,
              )
            }
            placeholder="Type exactly what is said in the clip."
            className="min-h-36 bg-card leading-6 shadow-none"
          />
          <p className="-mt-2 text-xs text-muted-foreground">
            An accurate transcript helps the model match the voice.
          </p>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => setPendingReference(null)}>
              Cancel
            </Button>
            <Button type="button" onClick={savePendingReference}>
              {pendingReference?.mode === 'create' ? 'Add clip' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default App
