import { useEffect, useRef, useState } from 'react'
import {
  AudioLines,
  ChevronDown,
  CircleAlert,
  Copy,
  Download,
  FileText,
  Info,
  LoaderCircle,
  Plus,
  Settings2,
  Upload,
} from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'

type AudioFormat = 'mp3' | 'wav' | 'pcm' | 'opus'
type LatencyMode = 'low' | 'normal'

const defaultInputText = `[excited, joyful tone] We're going to DISNEY WORLD! [squeal of delight] I've been saving for [emphasis] three years [breathless] and finally, FINALLY we can go! The look on your face right now is worth every extra shift I worked!
[angry] After everything we've been through [break] I can't believe you would [emphasize] betray me like this. I gave you EVERYTHING! And now I'm left with nothing but memories and broken promises!`


type ControlsState = {
  chunkLength: number
  maxNewTokens: number
  temperature: number
  topP: number
  repetitionPenalty: number
  normalize: boolean
  format: AudioFormat
  latency: LatencyMode
}

type Metrics = {
  textLength: number
  ttftMs: number
  receivedKb: number
}

type StatusState = {
  tone: 'error' | 'info'
  message: string
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
  audio?: ArrayBuffer
  text: string
}

const initialControls: ControlsState = {
  chunkLength: 1000,
  maxNewTokens: 2048,
  temperature: 0.9,
  topP: 0.9,
  repetitionPenalty: 1.05,
  normalize: false,
  format: 'mp3',
  latency: 'normal',
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

function createSpeakerGroup(): SpeakerGroup {
  return {
    id: createId(),
    references: [],
  }
}

const initialSpeakerGroup = createSpeakerGroup()

function buildReferencesPayload(
  speakerGroups: SpeakerGroup[],
  includeBinaryAudio: boolean,
) {
  const groupedReferences = speakerGroups
    .map((speakerGroup) =>
      speakerGroup.references.map((reference) => ({
        text: reference.text,
        audio: includeBinaryAudio
          ? Array.from(new Uint8Array(reference.audio))
          : '<audio binary data>',
      })),
    )
    .filter((speakerReferences) => speakerReferences.length > 0)

  if (groupedReferences.length <= 1) {
    return groupedReferences[0] ?? []
  }

  return groupedReferences
}

function buildPreviewPayload(
  inputText: string,
  controls: ControlsState,
  speakerGroups: SpeakerGroup[],
) {
  return {
    text: inputText,
    chunk_length: controls.chunkLength,
    max_new_tokens: controls.maxNewTokens,
    format: controls.format,
    latency: controls.latency,
    normalize: controls.normalize,
    references: buildReferencesPayload(speakerGroups, false),
    temperature: controls.temperature,
    top_p: controls.topP,
    repetition_penalty: controls.repetitionPenalty,
  }
}

function buildRequestPayload(
  inputText: string,
  controls: ControlsState,
  speakerGroups: SpeakerGroup[],
) {
  return {
    text: inputText,
    chunk_length: controls.chunkLength,
    max_new_tokens: controls.maxNewTokens,
    format: controls.format,
    latency: controls.latency,
    normalize: controls.normalize,
    references: buildReferencesPayload(speakerGroups, true),
    temperature: controls.temperature,
    top_p: controls.topP,
    repetition_penalty: controls.repetitionPenalty,
  }
}

function createFileName(inputText: string) {
  const safePrefix = inputText.trim().replace(/\s+/g, '-').slice(0, 24) || 'tts'
  return safePrefix
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Unknown error'
}

function waitForSourceBuffer(sourceBuffer: SourceBuffer) {
  if (!sourceBuffer.updating) {
    return Promise.resolve()
  }

  return new Promise<void>((resolve) => {
    const handleUpdateEnd = () => {
      sourceBuffer.removeEventListener('updateend', handleUpdateEnd)
      resolve()
    }

    sourceBuffer.addEventListener('updateend', handleUpdateEnd)
  })
}

function canUseStreamingPlayback(format: AudioFormat) {
  const mime = formatMimeMap[format]
  return typeof window.MediaSource !== 'undefined' && MediaSource.isTypeSupported(mime)
}

type SettingSliderProps = {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onValueChange: (value: number) => void
  formatValue?: (value: number) => string
}

function SettingSlider({
  label,
  value,
  min,
  max,
  step = 1,
  onValueChange,
  formatValue,
}: SettingSliderProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-4">
        <Label>{label}</Label>
        <span className="text-sm text-muted-foreground">
          {formatValue ? formatValue(value) : value}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(nextValue) => {
          const current = nextValue[0]
          if (typeof current === 'number') {
            onValueChange(current)
          }
        }}
      />
    </div>
  )
}

function App() {
  const [inputText, setInputText] = useState(defaultInputText)
  const [controls, setControls] = useState(initialControls)
  const [speakerGroups, setSpeakerGroups] = useState<SpeakerGroup[]>([initialSpeakerGroup])
  const [pendingReference, setPendingReference] = useState<PendingReference | null>(null)
  const [openSpeakerIds, setOpenSpeakerIds] = useState<number[]>([initialSpeakerGroup.id])
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [copyLabel, setCopyLabel] = useState('Copy')
  const [isRequestPreviewOpen, setIsRequestPreviewOpen] = useState(false)
  const [statusMessage, setStatusMessage] = useState<StatusState | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [downloadName, setDownloadName] = useState('generated-audio.mp3')

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const speakerGroupsRef = useRef<SpeakerGroup[]>([])
  const uploadTargetSpeakerIdRef = useRef<number | null>(null)
  const downloadUrlRef = useRef<string | null>(null)
  const mediaSourceUrlRef = useRef<string | null>(null)

  speakerGroupsRef.current = speakerGroups

  useEffect(() => {
    return () => {
      speakerGroupsRef.current.forEach((speakerGroup) => {
        speakerGroup.references.forEach((reference) => {
          URL.revokeObjectURL(reference.previewUrl)
        })
      })

      if (downloadUrlRef.current) {
        URL.revokeObjectURL(downloadUrlRef.current)
      }

      if (mediaSourceUrlRef.current) {
        URL.revokeObjectURL(mediaSourceUrlRef.current)
      }
    }
  }, [])

  function addSpeaker() {
    const nextSpeaker = createSpeakerGroup()
    setSpeakerGroups((current) => [...current, nextSpeaker])
    setOpenSpeakerIds((current) => [...current, nextSpeaker.id])
  }

  function removeSpeaker(speakerId: number) {
    setSpeakerGroups((current) => {
      const targetSpeaker = current.find((speakerGroup) => speakerGroup.id === speakerId)
      if (targetSpeaker) {
        targetSpeaker.references.forEach((reference) => {
          URL.revokeObjectURL(reference.previewUrl)
        })
      }

      const next = current.filter((speakerGroup) => speakerGroup.id !== speakerId)
      return next.length > 0 ? next : [createSpeakerGroup()]
    })
    setOpenSpeakerIds((current) => current.filter((currentSpeakerId) => currentSpeakerId !== speakerId))

    if (pendingReference?.speakerId === speakerId) {
      setPendingReference(null)
    }
  }

  function addReference(speakerId: number, name: string, audio: ArrayBuffer, text: string) {
    const previewUrl = URL.createObjectURL(new Blob([audio], { type: formatMimeMap.mp3 }))

    setSpeakerGroups((current) =>
      current.map((speakerGroup) =>
        speakerGroup.id === speakerId
          ? {
              ...speakerGroup,
              references: [
                ...speakerGroup.references,
                {
                  id: createId(),
                  name,
                  audio,
                  text,
                  previewUrl,
                },
              ],
            }
          : speakerGroup,
      ),
    )
  }

  function removeReference(speakerId: number, referenceId: number) {
    setSpeakerGroups((current) =>
      current.map((speakerGroup) => {
        if (speakerGroup.id !== speakerId) {
          return speakerGroup
        }

        return {
          ...speakerGroup,
          references: speakerGroup.references.filter((reference) => {
            if (reference.id === referenceId) {
              URL.revokeObjectURL(reference.previewUrl)
              return false
            }

            return true
          }),
        }
      }),
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

  function clearDownloadUrl() {
    if (downloadUrlRef.current) {
      URL.revokeObjectURL(downloadUrlRef.current)
      downloadUrlRef.current = null
    }

    setDownloadUrl(null)
  }

  function clearMediaSourceUrl() {
    if (mediaSourceUrlRef.current) {
      URL.revokeObjectURL(mediaSourceUrlRef.current)
      mediaSourceUrlRef.current = null
    }
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
        pendingReference.text,
      )
    }

    if (pendingReference.mode === 'edit' && typeof pendingReference.referenceId === 'number') {
      updateReferenceText(
        pendingReference.speakerId,
        pendingReference.referenceId,
        pendingReference.text,
      )
    }

    setPendingReference(null)
    setStatusMessage(null)
  }

  async function copyRequestPreview() {
    const requestPreview = JSON.stringify(
      buildPreviewPayload(inputText, controls, speakerGroups),
      null,
      2,
    )

    try {
      await navigator.clipboard.writeText(requestPreview)
      setCopyLabel('Copied')
      window.setTimeout(() => setCopyLabel('Copy'), 2000)
    } catch (error) {
      setStatusMessage({
        tone: 'error',
        message: `Failed to copy request preview: ${getErrorMessage(error)}`,
      })
    }
  }

  async function handleGenerateAudio() {
    const audioElement = audioRef.current
    if (!audioElement) {
      return
    }

    const mime = formatMimeMap[controls.format]
    const useStreamingPlayback = canUseStreamingPlayback(controls.format)

    clearDownloadUrl()
    clearMediaSourceUrl()
    setMetrics(null)
    setStatusMessage(null)
    setIsGenerating(true)

    try {
      const response = await fetch('/v1/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(buildRequestPayload(inputText, controls, speakerGroups)),
      })

      if (!response.ok || !response.body) {
        throw new Error('Failed to generate audio')
      }

      const reader = response.body.getReader()
      let mediaSource: MediaSource | null = null

      if (useStreamingPlayback) {
        mediaSource = new MediaSource()
        const streamUrl = URL.createObjectURL(mediaSource)
        mediaSourceUrlRef.current = streamUrl
        audioElement.src = streamUrl
      } else {
        audioElement.removeAttribute('src')
        audioElement.load()
      }

      const allChunks: ArrayBuffer[] = []
      const playQueue: ArrayBuffer[] = []
      let sourceBuffer: SourceBuffer | null = null
      let readingDone = false
      let receivedLength = 0
      let ttftMs = -1
      const startTime = performance.now()

      if (mediaSource) {
        const sourceReady = new Promise<void>((resolve, reject) => {
          mediaSource.addEventListener(
            'sourceopen',
            () => {
              try {
                sourceBuffer = mediaSource.addSourceBuffer(mime)

                const processQueue = async () => {
                  if (!sourceBuffer || !mediaSource) {
                    return
                  }

                  while (true) {
                    if (readingDone && playQueue.length === 0) {
                      await waitForSourceBuffer(sourceBuffer)
                      if (mediaSource.readyState === 'open') {
                        mediaSource.endOfStream()
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

        await sourceReady
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          readingDone = true
          break
        }

        receivedLength += value.byteLength

        if (ttftMs < 0) {
          ttftMs = performance.now() - startTime
        }

        setMetrics({
          textLength: inputText.length,
          ttftMs,
          receivedKb: Math.round(receivedLength / 1024),
        })

        const chunk = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)
        playQueue.push(chunk)
        allChunks.push(chunk)

        if (useStreamingPlayback && audioElement.paused) {
          void audioElement.play().catch(() => undefined)
        }
      }

      const audioBlob = new Blob(allChunks, { type: mime })
      const nextDownloadUrl = URL.createObjectURL(audioBlob)
      downloadUrlRef.current = nextDownloadUrl
      setDownloadUrl(nextDownloadUrl)
      setDownloadName(`${createFileName(inputText)}.${controls.format}`)

      if (!useStreamingPlayback) {
        audioElement.src = nextDownloadUrl
        audioElement.load()
        setStatusMessage({
          tone: 'info',
          message: `Format "${controls.format}" is not supported for in-browser playback. The file is ready to download after generation completes.`,
        })
      }
    } catch (error) {
      setStatusMessage({
        tone: 'error',
        message: `Audio generation failed: ${getErrorMessage(error)}`,
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const requestPreview = JSON.stringify(
    buildPreviewPayload(inputText, controls, speakerGroups),
    null,
    2,
  )

  const totalReferenceCount = speakerGroups.reduce(
    (count, speakerGroup) => count + speakerGroup.references.length,
    0,
  )

  return (
    <main className="min-h-screen bg-zinc-50">
      <div className="mx-auto max-w-[1600px] px-3 py-3 sm:px-4 lg:px-5">
        <div className="grid gap-4 xl:h-[calc(100vh-1.5rem)] xl:grid-cols-[minmax(0,1fr)_460px]">
          <section className="grid gap-4 xl:min-h-0 xl:grid-rows-[minmax(0,1fr)_auto]">
            <Card className="rounded-xl border-zinc-200 bg-white shadow-none xl:min-h-0 xl:flex xl:flex-col">
              <CardHeader className="space-y-1 border-b border-zinc-100 px-4 py-4">
                <div className="flex items-center gap-2 text-zinc-700">
                  <FileText className="size-4" />
                  <CardTitle>Input</CardTitle>
                </div>
                <CardDescription>
                  Enter the text to synthesize and inspect the outgoing request payload.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 px-4 pt-4 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
                <div className="space-y-2">
                  <Label htmlFor="inputText">Input Text</Label>
                  <Textarea
                    id="inputText"
                    value={inputText}
                    onChange={(event) => setInputText(event.target.value)}
                    placeholder="Enter text to synthesize"
                    className="min-h-[220px] resize-y rounded-xl border-zinc-200 bg-white p-3 text-sm shadow-none focus-visible:ring-zinc-300 xl:min-h-[260px]"
                  />
                </div>

                <Collapsible open={isRequestPreviewOpen} onOpenChange={setIsRequestPreviewOpen}>
                  <div className="rounded-xl border border-zinc-200 bg-zinc-50">
                    <div className="flex flex-col gap-2 p-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <div className="text-sm font-medium text-zinc-900">Request Preview</div>
                        <div className="text-xs text-zinc-500">
                          Live snapshot of the payload sent to the backend.
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="border border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-100"
                          onClick={copyRequestPreview}
                        >
                          <Copy className="size-3.5" />
                          {copyLabel}
                        </Button>
                        <CollapsibleTrigger asChild>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="border border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-100"
                          >
                            {isRequestPreviewOpen ? 'Collapse' : 'Expand'}
                            <ChevronDown
                              className={`size-4 transition-transform ${
                                isRequestPreviewOpen ? 'rotate-180' : ''
                              }`}
                            />
                          </Button>
                        </CollapsibleTrigger>
                      </div>
                    </div>
                    <CollapsibleContent>
                      <Separator className="bg-zinc-200" />
                      <div className="p-3 pt-3">
                        <ScrollArea className="h-56 min-w-0 rounded-lg border border-zinc-200 bg-white">
                          <pre className="max-w-full whitespace-pre-wrap break-all p-3 text-xs leading-5 text-zinc-700">
                            {requestPreview}
                          </pre>
                        </ScrollArea>
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>

                <div className="space-y-4">
                  <Button
                    type="button"
                    size="lg"
                    className="h-11 rounded-lg bg-zinc-900 text-white hover:bg-zinc-800"
                    onClick={handleGenerateAudio}
                    disabled={isGenerating}
                  >
                    {isGenerating ? (
                      <LoaderCircle className="size-4 animate-spin" />
                    ) : (
                      <AudioLines className="size-4" />
                    )}
                    {isGenerating ? 'Generating Audio...' : 'Generate Audio'}
                  </Button>

                  {statusMessage ? (
                    <Alert
                      variant={statusMessage.tone === 'error' ? 'destructive' : 'warning'}
                      className="rounded-lg"
                    >
                      <div className="flex items-start gap-3">
                        {statusMessage.tone === 'error' ? (
                          <CircleAlert className="mt-0.5 size-4 shrink-0" />
                        ) : (
                          <Info className="mt-0.5 size-4 shrink-0" />
                        )}
                        <div>
                          <AlertTitle>
                            {statusMessage.tone === 'error' ? 'Error' : 'Notice'}
                          </AlertTitle>
                          <AlertDescription>{statusMessage.message}</AlertDescription>
                        </div>
                      </div>
                    </Alert>
                  ) : null}
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-xl border-zinc-200 bg-white shadow-none">
              <CardHeader className="space-y-1 border-b border-zinc-100 px-4 py-4">
                <div className="flex items-center gap-2 text-zinc-700">
                  <AudioLines className="size-4" />
                  <CardTitle>Output</CardTitle>
                </div>
                <CardDescription>
                  Stream the result when supported, then preview or download the final file.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 px-4 pt-4">
                <audio
                  ref={audioRef}
                  controls
                  className="w-full rounded-lg border border-zinc-200 bg-white"
                />

                <div className="flex flex-wrap gap-2">
                  {metrics ? (
                    <>
                      <Badge variant="outline" className="border-zinc-200 bg-white text-zinc-700">
                        Text length: {metrics.textLength}
                      </Badge>
                      <Badge variant="outline" className="border-zinc-200 bg-white text-zinc-700">
                        TTFT: {metrics.ttftMs.toFixed(2)} ms
                      </Badge>
                      <Badge variant="outline" className="border-zinc-200 bg-white text-zinc-700">
                        Received: {metrics.receivedKb} KB
                      </Badge>
                    </>
                  ) : (
                    <Badge variant="outline" className="border-zinc-200 bg-white text-zinc-500">
                      No output yet
                    </Badge>
                  )}
                </div>

                <div className="flex justify-end">
                  {downloadUrl ? (
                    <Button
                      asChild
                      variant="outline"
                      className="border-zinc-200 bg-white text-zinc-800 hover:bg-zinc-100"
                    >
                      <a href={downloadUrl} download={downloadName}>
                        <Download className="size-4" />
                        Download
                      </a>
                    </Button>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          </section>

          <aside className="grid gap-4 xl:min-h-0 xl:grid-rows-[minmax(0,1fr)_auto]">
            <Card className="rounded-xl border-zinc-200 bg-white shadow-none xl:min-h-0 xl:flex xl:flex-col">
              <CardHeader className="space-y-1 border-b border-zinc-100 px-4 py-4">
                <div className="flex items-center gap-2 text-zinc-700">
                  <Upload className="size-4" />
                  <CardTitle>Reference Audio</CardTitle>
                </div>
                <CardDescription>
                  Build one or more speaker groups. Each speaker can have multiple reference clips.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 px-4 pt-4 xl:min-h-0 xl:flex xl:flex-1 xl:flex-col">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center text-sm text-zinc-500">
                    {speakerGroups.length} speaker{speakerGroups.length === 1 ? '' : 's'} /{' '}
                    {totalReferenceCount} reference{totalReferenceCount === 1 ? '' : 's'}
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    className="border-zinc-200 bg-white hover:bg-zinc-100"
                    onClick={addSpeaker}
                  >
                    <Plus className="size-4" />
                    Add Speaker
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={handleReferenceUpload}
                  />
                </div>

                <ScrollArea className="min-h-0 rounded-md xl:h-full xl:flex-1">
                  <div className="space-y-2">
                    {speakerGroups.length > 0 ? (
                      speakerGroups.map((speakerGroup, speakerIndex) => (
                        <Collapsible
                          key={speakerGroup.id}
                          open={openSpeakerIds.includes(speakerGroup.id)}
                          onOpenChange={(open) => {
                            setOpenSpeakerIds((current) =>
                              open
                                ? [...current, speakerGroup.id]
                                : current.filter(
                                    (currentSpeakerId) => currentSpeakerId !== speakerGroup.id,
                                  ),
                            )
                          }}
                        >
                          <div className="rounded-lg border border-zinc-200 bg-white">
                            <div className="flex flex-col gap-2 px-3 py-3 sm:flex-row sm:items-center sm:justify-between">
                              <div className="min-w-0">
                                <div className="text-sm font-medium text-zinc-900">
                                  Speaker {speakerIndex}
                                </div>
                                <div className="text-xs text-zinc-500">
                                  {speakerGroup.references.length} reference
                                  {speakerGroup.references.length === 1 ? '' : 's'}
                                </div>
                              </div>
                              <div className="flex flex-wrap gap-2">
                                <Button
                                  type="button"
                                  variant="outline"
                                  size="sm"
                                  className="h-8 border-zinc-200 bg-white px-2.5 hover:bg-zinc-100"
                                  onClick={() => {
                                    uploadTargetSpeakerIdRef.current = speakerGroup.id
                                    fileInputRef.current?.click()
                                  }}
                                >
                                  <Upload className="size-4" />
                                  Upload
                                </Button>
                                {speakerGroups.length > 1 ? (
                                  <Button
                                    type="button"
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 px-2.5 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
                                    onClick={() => removeSpeaker(speakerGroup.id)}
                                  >
                                    Remove
                                  </Button>
                                ) : null}
                                <CollapsibleTrigger asChild>
                                  <Button
                                    type="button"
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 px-2 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
                                  >
                                    <ChevronDown
                                      className={`size-4 transition-transform ${
                                        openSpeakerIds.includes(speakerGroup.id) ? 'rotate-180' : ''
                                      }`}
                                    />
                                  </Button>
                                </CollapsibleTrigger>
                              </div>
                            </div>

                            <CollapsibleContent>
                              <Separator className="bg-zinc-200" />
                              <div className="space-y-2 px-3 py-2.5">
                                {speakerGroup.references.length > 0 ? (
                                  speakerGroup.references.map((reference) => (
                                    <div
                                      key={reference.id}
                                      className="flex flex-col gap-2 rounded-md border border-zinc-200 bg-zinc-50 p-2 sm:flex-row sm:items-center"
                                    >
                                      <audio
                                        controls
                                        src={reference.previewUrl}
                                        className="h-9 w-full min-w-0 rounded-md border border-zinc-200 bg-white sm:flex-1"
                                      />
                                      <div className="flex gap-2 sm:shrink-0">
                                        <Button
                                          type="button"
                                          variant="ghost"
                                          size="sm"
                                          className="h-8 border border-zinc-200 bg-white px-2.5 text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
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
                                          Edit Text
                                        </Button>
                                        <Button
                                          type="button"
                                          variant="ghost"
                                          size="sm"
                                          className="h-8 border border-zinc-200 bg-white px-2.5 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
                                          onClick={() =>
                                            removeReference(speakerGroup.id, reference.id)
                                          }
                                        >
                                          Remove
                                        </Button>
                                      </div>
                                    </div>
                                  ))
                                ) : (
                                  <div className="px-1 py-3 text-sm text-zinc-500">
                                    No references yet.
                                  </div>
                                )}
                              </div>
                            </CollapsibleContent>
                          </div>
                        </Collapsible>
                      ))
                    ) : (
                      <div className="rounded-lg border border-dashed border-zinc-300 bg-white p-4 text-sm text-zinc-500">
                        No speaker groups configured yet.
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            <Card className="rounded-xl border-zinc-200 bg-white shadow-none">
              <CardHeader className="space-y-1 border-b border-zinc-100 px-4 py-4">
                <div className="flex items-center gap-2 text-zinc-700">
                  <Settings2 className="size-4" />
                  <CardTitle>Generation Settings</CardTitle>
                </div>
                <CardDescription>Adjust sampling and output parameters.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 px-4 pt-4">
                <div className="space-y-2">
                  <Label>Latency Mode</Label>
                  <ToggleGroup
                    type="single"
                    value={controls.latency}
                    className="grid grid-cols-2 gap-2"
                    onValueChange={(value) => {
                      if (value) {
                        setControls((current) => ({
                          ...current,
                          latency: value as LatencyMode,
                        }))
                      }
                    }}
                  >
                    <ToggleGroupItem value="low" className="w-full">
                      low
                    </ToggleGroupItem>
                    <ToggleGroupItem value="normal" className="w-full">
                      normal
                    </ToggleGroupItem>
                  </ToggleGroup>
                  <p className="text-xs text-zinc-500">
                    Low uses incremental local decode for faster first audio. Normal waits for the
                    full LLM result, then decodes once.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Format</Label>
                  <ToggleGroup
                    type="single"
                    value={controls.format}
                    className="grid grid-cols-4 gap-2"
                    onValueChange={(value) => {
                      if (value) {
                        setControls((current) => ({
                          ...current,
                          format: value as AudioFormat,
                        }))
                      }
                    }}
                  >
                    <ToggleGroupItem value="mp3" className="w-full">
                      mp3
                    </ToggleGroupItem>
                    <ToggleGroupItem value="wav" className="w-full">
                      wav
                    </ToggleGroupItem>
                    <ToggleGroupItem value="pcm" className="w-full">
                      pcm
                    </ToggleGroupItem>
                    <ToggleGroupItem value="opus" className="w-full">
                      opus
                    </ToggleGroupItem>
                  </ToggleGroup>
                </div>

                <div className="flex items-center justify-between rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5">
                  <div className="space-y-1">
                    <Label htmlFor="normalize">Normalize</Label>
                    <p className="text-xs text-zinc-500">
                      Normalize text before synthesis to keep input formatting consistent.
                    </p>
                  </div>
                  <Switch
                    id="normalize"
                    checked={controls.normalize}
                    onCheckedChange={(checked) =>
                      setControls((current) => ({
                        ...current,
                        normalize: checked,
                      }))
                    }
                  />
                </div>

                <Separator className="bg-zinc-200" />

                <SettingSlider
                  label="Chunk Length"
                  value={controls.chunkLength}
                  min={100}
                  max={1000}
                  onValueChange={(value) =>
                    setControls((current) => ({
                      ...current,
                      chunkLength: value,
                    }))
                  }
                />
                <SettingSlider
                  label="Max New Tokens"
                  value={controls.maxNewTokens}
                  min={256}
                  max={2048}
                  onValueChange={(value) =>
                    setControls((current) => ({
                      ...current,
                      maxNewTokens: value,
                    }))
                  }
                />
                <SettingSlider
                  label="Temperature"
                  value={controls.temperature}
                  min={0.8}
                  max={1}
                  step={0.01}
                  formatValue={(value) => value.toFixed(2)}
                  onValueChange={(value) =>
                    setControls((current) => ({
                      ...current,
                      temperature: value,
                    }))
                  }
                />
                <SettingSlider
                  label="Top P"
                  value={controls.topP}
                  min={0.8}
                  max={1}
                  step={0.01}
                  formatValue={(value) => value.toFixed(2)}
                  onValueChange={(value) =>
                    setControls((current) => ({
                      ...current,
                      topP: value,
                    }))
                  }
                />
                <SettingSlider
                  label="Repetition Penalty"
                  value={controls.repetitionPenalty}
                  min={1}
                  max={1.2}
                  step={0.01}
                  formatValue={(value) => value.toFixed(2)}
                  onValueChange={(value) =>
                    setControls((current) => ({
                      ...current,
                      repetitionPenalty: value,
                    }))
                  }
                />
              </CardContent>
            </Card>
          </aside>
        </div>
      </div>

      <Dialog open={pendingReference !== null} onOpenChange={(open) => !open && setPendingReference(null)}>
        <DialogContent className="border-zinc-200 bg-white">
          <DialogHeader>
            <DialogTitle>
              {pendingReference?.mode === 'create' ? 'Save Reference Text' : 'Edit Reference Text'}
            </DialogTitle>
            <DialogDescription>
              {pendingReference
                ? `Speaker ${speakerGroups.findIndex(
                    (speakerGroup) => speakerGroup.id === pendingReference.speakerId,
                  )}`
                : ''}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="text-sm font-medium text-zinc-900">{pendingReference?.name}</div>
            <Textarea
              value={pendingReference?.text ?? ''}
              onChange={(event) =>
                setPendingReference((current) =>
                  current
                    ? {
                        ...current,
                        text: event.target.value,
                      }
                    : current,
                )
              }
              placeholder="Enter reference text"
              className="min-h-40 rounded-lg border-zinc-200 bg-white shadow-none focus-visible:ring-zinc-300"
            />
          </div>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => setPendingReference(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              variant="outline"
              className="border-zinc-200 bg-white hover:bg-zinc-100"
              onClick={savePendingReference}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  )
}

export default App
