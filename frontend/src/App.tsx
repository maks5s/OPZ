import { ChangeEvent, ReactNode, useEffect, useRef, useState } from "react"
import JSZip from "jszip"

type IconName = "adjust" | "archive" | "check" | "close" | "download" | "image" | "play" | "settings" | "spark" | "trash" | "upload"

type GeneratedImage = {
  id: number
  name: string
  src: string
}

type SystemSettings = {
  random_seed: number | null
  img_size: number
  grid_cells: number
}

type TrainingSettings = {
  epochs: number
  learning_rate: number
  hidden_layers: 0 | 1
  hidden_units: number
}

type PerClassMetrics = {
  precision: number
  recall: number
  f1: number
  support: number
}

type ModelMetrics = {
  train_samples: number
  test_samples: number
  train_accuracy: number
  test_accuracy: number
  train_loss: number
  test_loss: number
  test_balanced_accuracy?: number
  test_precision?: number
  test_recall?: number
  test_f1?: number
  test_roc_auc?: number | null
  test_confusion_matrix?: number[][]
  per_class?: {
    class0: PerClassMetrics
    class1: PerClassMetrics
  }
  class_distribution?: {
    class0: number
    class1: number
  }
  class_weights?: {
    class0: number
    class1: number
  }
  epochs: number
  learning_rate?: number
  hidden_layers?: number
  hidden_units?: number
  feature_count?: number
  threshold?: number
  duration_seconds?: number
  trained_at?: string
}

type ModelInfo = {
  format_version?: number
  framework?: string
  architecture?: string
  img_size: number
  grid_cells: number
  feature_count: number
  hidden_layers?: number
  hidden_units?: number
  hidden_activation?: string
  optimizer?: string
  epochs: number
  learning_rate: number
  random_state: number | null
  threshold: number
  trained_at: string
  duration_seconds: number
  source?: string
  dataset_filename?: string | null
}

type TrainingProgress = {
  current: number
  total: number
  train_accuracy?: number
  test_accuracy?: number
}

type ModelStatus = {
  ready: boolean
  training: boolean
  status: "ready" | "training" | "stale" | "not_ready"
  message: string
  model: ModelInfo | null
  metrics: ModelMetrics | null
  progress: TrainingProgress | null
  settings: {
    img_size: number
    grid_cells: number
  }
}

type Prediction = {
  has_defect: boolean
  label: "defect" | "clean"
  label_uk: string
  probability: number
  confidence: number
  threshold: number
}

type LogMessage = {
  type: string
  text: string
  timestamp: string
  current?: number
  total?: number
  train_accuracy?: number
  test_accuracy?: number
  metrics?: ModelMetrics
  model?: ModelInfo
  threshold?: number
}

const API_BASE_URL = "http://127.0.0.1:8000"
const IMAGE_EXTENSION_REGEX = /\.(png|jpe?g|bmp)$/i
const MAX_REPORTED_MISMATCHES = 5

type Dimensions = {
  width: number
  height: number
}

type AnalysisResult = {
  difference_grid: string
  source_with_grid: string
  grid_cells: number
  img_size: number
  model_ready: boolean
  prediction: Prediction | null
  prediction_message: string | null
}

function base64ToBlob(base64: string, mimeType: string) {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return new Blob([bytes], { type: mimeType })
}

async function readImageDimensions(blob: Blob): Promise<Dimensions> {
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob)
    const dimensions = { width: bitmap.width, height: bitmap.height }
    bitmap.close()
    return dimensions
  }

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob)
    const image = new Image()

    image.onload = () => {
      resolve({ width: image.naturalWidth, height: image.naturalHeight })
      URL.revokeObjectURL(url)
    }
    image.onerror = () => {
      reject(new Error("Не вдалося прочитати зображення"))
      URL.revokeObjectURL(url)
    }
    image.src = url
  })
}

async function readErrorDetail(response: Response, fallback: string) {
  try {
    const data = (await response.json()) as { detail?: unknown }

    if (typeof data.detail === "string") return data.detail
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((item) => (typeof item === "string" ? item : JSON.stringify(item)))
        .join("; ")
    }
  } catch {
    // Відповідь без JSON — показуємо стандартне повідомлення.
  }

  return fallback
}

async function verifyArchiveDimensions(file: File, expectedSize: number) {
  let zip: JSZip

  try {
    zip = await JSZip.loadAsync(file)
  } catch {
    throw new Error("Не вдалося прочитати ZIP-архів")
  }

  const entries = Object.values(zip.files).filter(
    (entry) => !entry.dir && IMAGE_EXTENSION_REGEX.test(entry.name),
  )

  if (!entries.length) {
    throw new Error("В архіві не знайдено зображень PNG")
  }

  const mismatches: string[] = []
  let mismatchCount = 0

  for (const entry of entries) {
    try {
      const blob = await entry.async("blob")
      const { width, height } = await readImageDimensions(blob)

      if (width !== expectedSize || height !== expectedSize) {
        mismatchCount += 1
        if (mismatches.length < MAX_REPORTED_MISMATCHES) {
          mismatches.push(
            `${entry.name}: ${width}×${height}, очікується ${expectedSize}×${expectedSize}`,
          )
        }
      }
    } catch {
      mismatchCount += 1
      if (mismatches.length < MAX_REPORTED_MISMATCHES) {
        mismatches.push(`${entry.name}: не вдалося прочитати зображення`)
      }
    }
  }

  if (mismatchCount) {
    throw new Error(
      `Розмір зображень не відповідає налаштуванням (${expectedSize}×${expectedSize}). ` +
        `Проблемних файлів: ${mismatchCount} з ${entries.length}. ${mismatches.join("; ")}`,
    )
  }

  return entries.length
}

function getLogState(type: string) {
  if (type.includes("error")) return "warn"
  if (
    type.endsWith("_complete") ||
    type.endsWith("_updated") ||
    type.endsWith("_generated") ||
    type.endsWith("_loaded")
  ) {
    return "ok"
  }
  return "idle"
}

function formatLogTime(timestamp: string) {
  const date = new Date(timestamp)
  return Number.isNaN(date.getTime())
    ? "--:--:--"
    : date.toLocaleTimeString("uk-UA", { hour12: false })
}

function formatPercent(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(2)}%`
    : "—"
}

function formatDecimal(value?: number | null, digits = 4) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—"
}

function formatDateTime(value?: string | null) {
  if (!value) return "—"

  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("uk-UA", { hour12: false })
}

function formatModelSource(source?: string) {
  if (source === "trained") return "Навчена в системі"
  if (source === "loaded") return "Завантажена з файлу"
  return source || "—"
}

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    adjust: (
      <>
        <path d="M4 7h16M4 17h16" />
        <circle cx="9" cy="7" r="2" />
        <circle cx="15" cy="17" r="2" />
      </>
    ),
    archive: (
      <>
        <path d="M5 8h14v11H5zM4 4h16v4H4z" />
        <path d="M9 12h6" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    download: (
      <>
        <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 20h14" />
      </>
    ),
    image: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <circle cx="8.5" cy="9" r="1.5" />
        <path d="m4 17 5-5 3 3 2-2 6 6" />
      </>
    ),
    play: <path d="m9 7 8 5-8 5z" />,
    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2L12 3Z" />
        <path d="m6 14 .8 2.2L9 17l-2.2.8L6 20l-.8-2.2L3 17l2.2-.8L6 14Z" />
      </>
    ),
    trash: (
      <>
        <path d="M4 7h16M9 3h6l1 4H8l1-4ZM7 7l1 14h8l1-14M10 11v6M14 11v6" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m0 0L8 8m4-4 4 4M5 20h14" />
      </>
    ),
  }

  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.8"
    >
      {paths[name]}
    </svg>
  )
}

function Modal({
  children,
  onClose,
  title,
}: {
  children: ReactNode
  onClose: () => void
  title: string
}) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose()
    }
    window.addEventListener("keydown", closeOnEscape)
    return () => window.removeEventListener("keydown", closeOnEscape)
  }, [onClose])

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div
        aria-modal="true"
        className="modal-panel"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <div className="modal-header">
          <div>
            <h2>{title}</h2>
          </div>
          <button
            aria-label="Закрити"
            className="icon-button"
            onClick={onClose}
          >
            <Icon name="close" />
          </button>
        </div>
        {children}
      </div>
    </div>
  )
}

function downloadData(url: string, filename: string) {
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.click()
}

function FileDrop({
  accept,
  file,
  label,
  onChange,
}: {
  accept: string
  file: File | null
  label: string
  onChange: (file: File | null) => void
}) {
  const id = `file-${label.replace(/\s/g, "-")}`
  return (
    <label className={`file-drop ${file ? "has-file" : ""}`} htmlFor={id}>
      <input
        accept={accept}
        id={id}
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
        type="file"
      />
      <span className="upload-icon">
        <Icon name={file ? "check" : "upload"} />
      </span>
      <span>
        <strong>{file ? file.name : label}</strong>
        <small>
          {file
            ? `${(file.size / 1024).toFixed(1)} КБ`
            : `Перетягніть сюди або оберіть файл`}
        </small>
      </span>
    </label>
  )
}

export default function App() {
  const [datasetTab, setDatasetTab] = useState<"generate" | "upload">("generate")
  const [modal, setModal] = useState<"reference-settings" | "train-settings" | "reference" | "dataset" | "source-grid" | "model-details" | null>(null)
  
  const [seed, setSeed] = useState<number | null>(null)
  const [imageSize, setImageSize] = useState(200)
  const [gridSize, setGridSize] = useState(10)
  const [reference, setReference] = useState("")
  const [datasetCount, setDatasetCount] = useState(250)
  const [dataset, setDataset] = useState<GeneratedImage[]>([])
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false)
  const [datasetProgress, setDatasetProgress] = useState({ current: 0, total: 0 })
  const [selectedDatasetImage, setSelectedDatasetImage] = useState<GeneratedImage | null>(null)
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [zipOrigin, setZipOrigin] = useState<"generated" | "uploaded" | null>(null)
  const [zipError, setZipError] = useState("")
  const [zipNotice, setZipNotice] = useState("")
  const [isValidatingZip, setIsValidatingZip] = useState(false)
  const [epochs, setEpochs] = useState(200)
  const [learningRate, setLearningRate] = useState(0.001)
  const [hiddenLayers, setHiddenLayers] = useState<0 | 1>(1)
  const [hiddenUnits, setHiddenUnits] = useState(16)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [isTraining, setIsTraining] = useState(false)
  const [modelReady, setModelReady] = useState(false)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null)
  const [trainingError, setTrainingError] = useState("")
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [isDownloadingModel, setIsDownloadingModel] = useState(false)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [modelError, setModelError] = useState("")
  const [modelNotice, setModelNotice] = useState("")
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [sourcePreview, setSourcePreview] = useState("")
  const [sourceError, setSourceError] = useState("")
  const [differencePreview, setDifferencePreview] = useState("")
  const [sourceGridPreview, setSourceGridPreview] = useState("")
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predictionMessage, setPredictionMessage] = useState("")
  const [processing, setProcessing] = useState(false)
  const [analysisError, setAnalysisError] = useState("")
  const [logs, setLogs] = useState<LogMessage[]>([])
  const logRef = useRef<HTMLDivElement>(null)
  const previewUrlsRef = useRef({
    sourcePreview: "",
    reference: "",
    differencePreview: "",
    sourceGridPreview: "",
    dataset: [] as string[],
  })
  const sourceValidationRef = useRef(0)
  const zipValidationRef = useRef(0)
  const analysisRequestRef = useRef(0)
  const [isGeneratingReference, setIsGeneratingReference] = useState(false)
  const [generationDuration, setGenerationDuration] = useState<number | null>(null)
  const [trainingDuration, setTrainingDuration] = useState<number | null>(null)

  

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${Math.round(ms)} мс`
    return `${(ms / 1000).toFixed(2)} с`
    }

  useEffect(() => {
    previewUrlsRef.current = {
      sourcePreview,
      reference,
      differencePreview,
      sourceGridPreview,
      dataset: dataset.map((image) => image.src),
    }
  }, [sourcePreview, reference, differencePreview, sourceGridPreview, dataset])

  useEffect(() => {
    return () => {
      const urlsToRevoke = [
        previewUrlsRef.current.sourcePreview,
        previewUrlsRef.current.reference,
        previewUrlsRef.current.differencePreview,
        previewUrlsRef.current.sourceGridPreview,
        ...previewUrlsRef.current.dataset,
      ]

      urlsToRevoke.forEach((url) => {
        if (url.startsWith("blob:")) URL.revokeObjectURL(url)
      })
    }
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [logs])

  useEffect(() => {
    const eventSource = new EventSource(`${API_BASE_URL}/logs/stream`)

    eventSource.onmessage = (event) => {
      let message: LogMessage

      try {
        message = JSON.parse(event.data) as LogMessage
      } catch {
        return
      }

      setLogs((current) => [...current, message])

      if (message.type === "dataset_started") {
        setDatasetProgress({
          current: message.current ?? 0,
          total: message.total ?? 0,
        })
      } else if (message.type === "dataset_progress") {
        setDatasetProgress({
          current: message.current ?? 0,
          total: message.total ?? 0,
        })
      } else if (message.type === "dataset_complete") {
        setDatasetProgress((current) => ({
          current: message.total ?? current.total,
          total: message.total ?? current.total,
        }))
      } else if (message.type === "model_training_started") {
        setIsTraining(true)
        setTrainingProgress(0)
        setTrainingError("")
        setTrainingDuration(null)
      } else if (message.type === "model_training_progress") {
        const current = message.current ?? 0
        const total = message.total ?? 0

        setIsTraining(true)
        setTrainingProgress(
          total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0,
        )
      } else if (message.type === "model_training_complete") {
        setIsTraining(false)
        setModelReady(true)
        setTrainingProgress(100)
        setTrainingError("")
        setModelError("")
        setModelNotice("")
        setPrediction(null)
        setPredictionMessage("")

        if (message.model) setModelInfo(message.model)

        if (message.metrics) {
          setModelMetrics(message.metrics)
          setTrainingDuration(
            message.metrics.duration_seconds
              ? message.metrics.duration_seconds * 1000
              : null,
          )
        }

        void loadModelStatus().catch(() => {
          // Помилки бекенду відображаються через SSE-логи.
        })
      } else if (message.type === "model_loaded") {
        setIsTraining(false)
        setModelReady(true)
        setTrainingProgress(100)
        setTrainingError("")
        setModelError("")
        setPrediction(null)
        setPredictionMessage("")

        if (message.model) setModelInfo(message.model)
        if (message.metrics) setModelMetrics(message.metrics)

        void loadModelStatus().catch(() => {
          // Помилки бекенду відображаються через SSE-логи.
        })
      } else if (message.type === "model_training_error") {
        setIsTraining(false)
        setTrainingError(message.text)
      } else if (message.type === "model_invalidated") {
        setModelReady(false)
        setModelInfo(null)
        setModelMetrics(null)
        setIsTraining(false)
        setTrainingProgress(0)
        setTrainingDuration(null)
        setTrainingError("")
        setPrediction(null)
        setPredictionMessage("")
        setModal((current) => (current === "model-details" ? null : current))
      }
    }

    return () => eventSource.close()
  }, [])

  const setReferenceFromBlob = (blob: Blob) => {
    const url = URL.createObjectURL(blob)
    setReference((current) => {
      if (current.startsWith("blob:")) URL.revokeObjectURL(current)
      return url
    })
  }

  const loadReference = async () => {
    const response = await fetch(`${API_BASE_URL}/template`, {
      cache: "no-store",
    })

    if (response.status === 404) {
      setReference((current) => {
        if (current.startsWith("blob:")) URL.revokeObjectURL(current)
        return ""
      })
      return false
    }

    if (!response.ok) throw new Error("Не вдалося отримати еталонне зображення")

    setReferenceFromBlob(await response.blob())
    return true
  }

  const applyModelStatus = (status: ModelStatus) => {
    setModelReady(status.ready)
    setModelInfo(status.model)
    setModelMetrics(status.metrics)
    setIsTraining(status.training)
    setTrainingError("")

    if (status.training) {
      const progress = status.progress
      setTrainingProgress(
        progress && progress.total > 0
          ? Math.min(100, Math.round((progress.current / progress.total) * 100))
          : 0,
      )
      return
    }

    if (status.ready) {
      setTrainingProgress(100)
      setTrainingDuration(
        status.metrics?.duration_seconds
          ? status.metrics.duration_seconds * 1000
          : null,
      )
    } else {
      setTrainingProgress(0)
      setTrainingDuration(null)
    }
  }

  const loadModelStatus = async () => {
    const response = await fetch(`${API_BASE_URL}/model/status`, {
      cache: "no-store",
    })

    if (!response.ok) throw new Error("Не вдалося отримати статус моделі")

    applyModelStatus((await response.json()) as ModelStatus)
  }

  useEffect(() => {
    let cancelled = false

    const loadBackendState = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/settings`)
        if (!response.ok) throw new Error("Не вдалося отримати налаштування системи")

        const settings = (await response.json()) as SystemSettings
        if (cancelled) return

        setSeed(settings.random_seed)
        setImageSize(settings.img_size)
        setGridSize(settings.grid_cells)
      } catch {
        // Помилки бекенду відображаються через SSE-логи.
      }

      if (cancelled) return

      try {
        const response = await fetch(`${API_BASE_URL}/training-settings`)
        if (!response.ok) throw new Error("Не вдалося отримати налаштування навчання")

        const settings = (await response.json()) as TrainingSettings
        if (cancelled) return

        setEpochs(settings.epochs)
        setLearningRate(settings.learning_rate)
        setHiddenLayers(settings.hidden_layers)
        setHiddenUnits(settings.hidden_units)
      } catch {
        // Помилки бекенду відображаються через SSE-логи.
      }

      if (cancelled) return

      try {
        await loadModelStatus()
      } catch {
        // Помилки бекенду відображаються через SSE-логи.
      }

      if (cancelled) return

      try {
        await loadReference()
      } catch {
        // Помилки бекенду відображаються через SSE-логи.
      }
    }

    void loadBackendState()

    return () => {
      cancelled = true
    }
  }, [])

  const clearAnalysisPreviews = () => {
    setDifferencePreview((current) => {
      if (current.startsWith("blob:")) URL.revokeObjectURL(current)
      return ""
    })
    setSourceGridPreview((current) => {
      if (current.startsWith("blob:")) URL.revokeObjectURL(current)
      return ""
    })
    setPrediction(null)
    setPredictionMessage("")
    setAnalysisError("")
  }

  const validateSourceFile = async (file: File, expectedSize: number) => {
    const validationId = ++sourceValidationRef.current

    try {
      const { width, height } = await readImageDimensions(file)
      if (validationId !== sourceValidationRef.current) return false

      if (width !== expectedSize || height !== expectedSize) {
        setSourceError(
          `Розмір ${width}×${height} не відповідає налаштуванням ${expectedSize}×${expectedSize}.`,
        )
        return false
      }

      setSourceError("")
      return true
    } catch {
      if (validationId === sourceValidationRef.current) {
        setSourceError("Не вдалося прочитати розміри зображення.")
      }
      return false
    }
  }

  const validateArchive = async (file: File, expectedSize: number) => {
    const validationId = ++zipValidationRef.current
    setIsValidatingZip(true)
    setZipError("")
    setZipNotice("")

    try {
      await verifyArchiveDimensions(file, expectedSize)
      if (validationId !== zipValidationRef.current) return

      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${API_BASE_URL}/dataset/validate`, {
        method: "POST",
        body: formData,
      })

      if (validationId !== zipValidationRef.current) return

      if (!response.ok) {
        throw new Error(
          await readErrorDetail(response, "Архів не пройшов перевірку розмірів"),
        )
      }

      const data = (await response.json()) as { images: number; img_size: number }
      if (validationId !== zipValidationRef.current) return

      setZipNotice(
        `Архів перевірено: ${data.images} зображень розміру ${data.img_size}×${data.img_size}.`,
      )
    } catch (error) {
      if (validationId !== zipValidationRef.current) return

      setZipError(
        error instanceof Error ? error.message : "Не вдалося перевірити ZIP-архів",
      )
    } finally {
      if (validationId === zipValidationRef.current) {
        setIsValidatingZip(false)
      }
    }
  }

  const handleZipUpload = (file: File | null) => {
    if (!file) {
      zipValidationRef.current += 1
      setIsValidatingZip(false)
      setZipFile(null)
      setZipOrigin(null)
      setZipError("")
      setZipNotice("")
      return
    }

    dataset.forEach((image) => {
      if (image.src.startsWith("blob:")) URL.revokeObjectURL(image.src)
    })
    setDataset([])
    setZipFile(file)
    setZipOrigin("uploaded")
    void validateArchive(file, imageSize)
  }

  const saveSettings = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          random_seed: seed,
          img_size: imageSize,
          grid_cells: gridSize,
        }),
      })

      if (!response.ok) return

      const data = await response.json()
      const currentSettings = data.current_settings as SystemSettings
      const imageSizeChanged = currentSettings.img_size !== imageSize

      setSeed(currentSettings.random_seed)
      setImageSize(currentSettings.img_size)
      setGridSize(currentSettings.grid_cells)

      setModal(null)
      analysisRequestRef.current += 1
      setProcessing(false)
      clearAnalysisPreviews()

      if (imageSizeChanged) {
        if (sourceFile) {
          void validateSourceFile(sourceFile, currentSettings.img_size)
        }

        dataset.forEach((image) => {
          if (image.src.startsWith("blob:")) URL.revokeObjectURL(image.src)
        })

        if (zipOrigin === "uploaded" && zipFile) {
          setDataset([])
          void validateArchive(zipFile, currentSettings.img_size)
        } else {
          zipValidationRef.current += 1
          setIsValidatingZip(false)
          setDataset([])
          setZipFile(null)
          setZipOrigin(null)
          setZipError("")
          setZipNotice("")
        }
      }

      try {
        await loadReference()
      } catch {
        // Помилки бекенду відображаються через SSE-логи.
      }

      try {
        await loadModelStatus()
      } catch {
        // Помилки бекенду відображаються через SSE-логи.
      }
    } catch {
      // Помилки бекенду відображаються через SSE-логи.
    }
  }

  const saveTrainingSettings = async () => {
    const safeHiddenUnits = Math.min(256, Math.max(1, Math.round(hiddenUnits) || 1))

    try {
      const response = await fetch(`${API_BASE_URL}/training-settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epochs,
          learning_rate: learningRate,
          hidden_layers: hiddenLayers,
          hidden_units: safeHiddenUnits,
        }),
      })

      if (!response.ok) {
        setTrainingError(
          await readErrorDetail(response, "Не вдалося зберегти налаштування навчання"),
        )
        return
      }

      const data = await response.json()
      const currentSettings = data.current_settings as TrainingSettings

      setEpochs(currentSettings.epochs)
      setLearningRate(currentSettings.learning_rate)
      setHiddenLayers(currentSettings.hidden_layers)
      setHiddenUnits(currentSettings.hidden_units)
      setTrainingError("")
      setModal(null)
    } catch {
      // Помилки бекенду відображаються через SSE-логи.
    }
  }

  const generateReference = async () => {
    if (isGeneratingReference) return
    setIsGeneratingReference(true)
    try {
      const response = await fetch(`${API_BASE_URL}/template`, {
        method: "POST",
      })
      if (!response.ok) throw new Error("Не вдалося згенерувати еталон")

      setReferenceFromBlob(await response.blob())
    } catch {
      // Помилки бекенду відображаються через SSE-логи.
    } finally {
      setIsGeneratingReference(false)
    }
  }

  const generateDataset = async () => {
    if (isGeneratingDataset) return
    zipValidationRef.current += 1
    setIsValidatingZip(false)
    setZipError("")
    setZipNotice("")
    setIsGeneratingDataset(true)
    setGenerationDuration(null)
    const startTime = performance.now()
    const count = Math.max(1, datasetCount || 1)
    setDatasetProgress({ current: 0, total: count })

    try {
      const response = await fetch(`${API_BASE_URL}/dataset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ total_samples: count }),
      })

      if (!response.ok) throw new Error("Помилка генерації на бекенді")

      const zipBlob = await response.blob()

      const zip = await JSZip.loadAsync(zipBlob)
      const images: GeneratedImage[] = []
      let id = 0

      for (const [filename, file] of Object.entries(zip.files)) {
        if (!file.dir && filename.endsWith(".png") && filename !== "template.png") {
          const fileBlob = await file.async("blob")
          images.push({
            id: id++,
            name: filename,
            src: URL.createObjectURL(fileBlob),
          })
        }
      }

      dataset.forEach((image) => {
        if (image.src.startsWith("blob:")) URL.revokeObjectURL(image.src)
      })

      setDataset(images)
      setZipFile(new File([zipBlob], "dataset.zip", { type: "application/zip" }))
      setZipOrigin("generated")
      setZipNotice(
        `Згенеровано ${images.length} зображень розміру ${imageSize}×${imageSize}.`,
      )
      setDatasetProgress({ current: count, total: count })
      const duration = performance.now() - startTime
      setGenerationDuration(duration)
    } catch {
      // Помилки бекенду відображаються через SSE-логи.
    } finally {
      setIsGeneratingDataset(false)
    }
  }

  const startTraining = async () => {
    if (!reference || !zipFile) return
    if (isValidatingZip || Boolean(zipError) || isTraining) return

    setIsTraining(true)
    setTrainingProgress(0)
    setTrainingError("")
    setModelError("")
    setModelNotice("")
    setTrainingDuration(null)
    const startTime = performance.now()

    try {
      const formData = new FormData()
      formData.append("file", zipFile)

      const response = await fetch(`${API_BASE_URL}/model/train`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const detail = await readErrorDetail(
          response,
          "Не вдалося навчити модель",
        )
        setTrainingError(detail)
        setIsTraining(false)
        return
      }

      const data = (await response.json()) as {
        metrics: ModelMetrics
        model: ModelInfo
      }

      setModelReady(true)
      setModelInfo(data.model)
      setModelMetrics(data.metrics)
      setTrainingProgress(100)
      setTrainingError("")
      setModelError("")
      setModelNotice("")
      setTrainingDuration(
        data.metrics.duration_seconds
          ? data.metrics.duration_seconds * 1000
          : performance.now() - startTime,
      )
    } catch {
      setTrainingError("Не вдалося отримати відповідь від сервера навчання")
    } finally {
      setIsTraining(false)
    }
  }

  const downloadModel = async () => {
    if (!modelReady || isDownloadingModel) return

    setIsDownloadingModel(true)
    setModelError("")
    setModelNotice("")

    try {
      const response = await fetch(`${API_BASE_URL}/model/download`, {
        cache: "no-store",
      })

      if (!response.ok) {
        throw new Error(
          await readErrorDetail(response, "Не вдалося скачати модель"),
        )
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      downloadData(url, "defect-model.zip")
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setModelNotice("Модель скачано у ZIP-архіві defect-model.zip.")
    } catch (error) {
      setModelError(
        error instanceof Error ? error.message : "Не вдалося скачати модель",
      )
    } finally {
      setIsDownloadingModel(false)
    }
  }

  const loadModelFile = async () => {
    if (!modelFile || isLoadingModel) return

    setIsLoadingModel(true)
    setModelError("")
    setModelNotice("")

    try {
      const formData = new FormData()
      formData.append("file", modelFile)

      const response = await fetch(`${API_BASE_URL}/model/load`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(
          await readErrorDetail(response, "Не вдалося завантажити модель"),
        )
      }

      const data = (await response.json()) as {
        model: ModelInfo
        metrics: ModelMetrics
      }

      setModelReady(true)
      setModelInfo(data.model)
      setModelMetrics(data.metrics)
      setTrainingProgress(100)
      setTrainingError("")
      setPrediction(null)
      setPredictionMessage("")
      setModelNotice("Модель успішно завантажено та готова до роботи.")

      try {
        await loadModelStatus()
      } catch {
        // Статус уже оновлено з відповіді /model/load.
      }
    } catch (error) {
      setModelError(
        error instanceof Error ? error.message : "Не вдалося завантажити модель",
      )
    } finally {
      setIsLoadingModel(false)
    }
  }

  const getTrainingHint = () => {
    if (!reference) return "Спочатку згенеруйте еталонне зображення"
    if (isTraining) return "Триває навчання моделі, епохи надходять у логи"
    if (modelReady) {
      return modelMetrics
        ? `Модель готова · test accuracy ${modelMetrics.test_accuracy}%`
        : "Модель готова до роботи"
    }
    if (isValidatingZip) return "Перевіряємо розміри зображень в архіві"
    if (zipError) return "Архів датасету не пройшов перевірку розмірів"
    if (!zipFile) return "Спочатку згенеруйте або завантажте датасет"

    const architecture =
      hiddenLayers === 1
        ? `1 прихований шар (${hiddenUnits} нейронів)`
        : "без прихованого шару"

    return `Дані готові · ${epochs} епох · LR ${learningRate} · ${architecture}`
  }

  const onSourceFile = (file: File | null) => {
    setSourceFile(file)
    sourceValidationRef.current += 1
    analysisRequestRef.current += 1
    setProcessing(false)
    setSourceError("")
    clearAnalysisPreviews()
    if (sourcePreview.startsWith("blob:")) URL.revokeObjectURL(sourcePreview)
    setSourcePreview(file ? URL.createObjectURL(file) : "")

    if (file) {
      void validateSourceFile(file, imageSize)
    }
  }

  const downloadDatasetZip = () => {
    if (!zipFile) return
    const url = URL.createObjectURL(zipFile)
    downloadData(url, "dataset.zip")
    URL.revokeObjectURL(url)
  }

  const downloadDifferenceGrid = () => {
    if (!differencePreview) return
    downloadData(differencePreview, "difference_grid.png")
  }

  const processImage = async () => {
    if (!sourceFile || processing) return

    const requestId = ++analysisRequestRef.current
    setProcessing(true)
    setAnalysisError("")
    clearAnalysisPreviews()

    try {
      const formData = new FormData()
      formData.append("file", sourceFile)

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      })

      if (requestId !== analysisRequestRef.current) return

      if (!response.ok) {
        const detail = await readErrorDetail(
          response,
          "Не вдалося сформувати різницеву сітку",
        )
        if (requestId !== analysisRequestRef.current) return

        setAnalysisError(detail)

        if (detail.includes("Розмір вхідного зображення")) {
          setSourceError(detail)
        }
        return
      }

      const result = (await response.json()) as AnalysisResult
      if (requestId !== analysisRequestRef.current) return

      const differenceBlob = base64ToBlob(result.difference_grid, "image/png")
      const sourceGridBlob = base64ToBlob(result.source_with_grid, "image/png")

      setDifferencePreview((current) => {
        if (current.startsWith("blob:")) URL.revokeObjectURL(current)
        return URL.createObjectURL(differenceBlob)
      })
      setSourceGridPreview((current) => {
        if (current.startsWith("blob:")) URL.revokeObjectURL(current)
        return URL.createObjectURL(sourceGridBlob)
      })
      setPrediction(result.prediction)
      setPredictionMessage(result.prediction_message ?? "")
    } catch {
      if (requestId === analysisRequestRef.current) {
        setAnalysisError("Не вдалося отримати результат аналізу")
      }
    } finally {
      if (requestId === analysisRequestRef.current) {
        setProcessing(false)
      }
    }
  }

  const datasetProgressPercent =
    datasetProgress.total > 0
      ? Math.min(100, Math.round((datasetProgress.current / datasetProgress.total) * 100))
      : 0

  const trainingReady =
    Boolean(zipFile) &&
    Boolean(reference) &&
    !isValidatingZip &&
    !zipError

  const displayedTrainingProgress = isTraining
    ? trainingProgress
    : modelReady
    ? 100
    : 0

  const currentFeatureCount = gridSize * gridSize
  const modelCompatible = Boolean(
    modelInfo &&
      modelInfo.img_size === imageSize &&
      modelInfo.grid_cells === gridSize &&
      modelInfo.feature_count === currentFeatureCount,
  )

  return (
    <div className="app-shell">
      <main>
        <section className="intro">
          <div>
            <h1>Розпізнавання кругів з дефектами</h1>
          </div>
        </section>

        <div className="workspace">
          <section className="zone">
            <div className="zone-heading">
              <span className="zone-number">01</span>
              <div>
                <h2>Керування</h2>
              </div>
            </div>

            <div className="reference-row">
              <button
                className="primary-button"
                disabled={isGeneratingReference}
                onClick={generateReference}
                >
                {isGeneratingReference ? (
                    <span className="button-spinner light" aria-hidden="true" />
                ) : (
                    <Icon name="spark" />
                )}
                {isGeneratingReference ? "Генерація..." : "Згенерувати еталон"}
                </button>
              {reference ? (
                <button
                  aria-label="Переглянути еталон"
                  className="reference-thumb"
                  onClick={() => setModal("reference")}
                >
                  <img alt="Згенерований еталон" src={reference} />
                  <span>
                    <Icon name="image" size={14} />
                  </span>
                </button>
              ) : (
                <span className="reference-hint">Еталон ще не створено</span>
              )}
              <button
                aria-label="Налаштування еталона"
                className="icon-button"
                onClick={() => setModal("reference-settings")}
              >
                <Icon name="settings" />
              </button>
            </div>

            <div className="tab-content">
              <div className="sub-tabs">
                  <button
                    className={datasetTab === "generate" ? "active" : ""}
                    onClick={() => setDatasetTab("generate")}
                  >
                    Генерація датасету
                  </button>
                  <button
                    className={datasetTab === "upload" ? "active" : ""}
                    onClick={() => setDatasetTab("upload")}
                  >
                    Завантаження датасету
                  </button>
                </div>

                {datasetTab === "generate" ? (
                  <div className="dataset-area">
                    <div className="inline-form">
                      <label>
                        <span>Кількість пар</span>
                        <input
                          max="500"
                          min="56"
                          onChange={(event) =>
                            setDatasetCount(Number(event.target.value))
                          }
                          type="number"
                          value={datasetCount}
                        />
                      </label>
                      <button
                        className="secondary-button"
                        disabled={isGeneratingDataset}
                        onClick={generateDataset}
                      >
                        <Icon name="spark" />
                        {isGeneratingDataset ? "Генерація..." : "Генерувати"}
                      </button>
                    </div>
                    {isGeneratingDataset && (
                      <div className="dataset-progress">
                        <div className="progress-label">
                          <span>
                            {datasetProgress.current} / {datasetProgress.total} пар
                          </span>
                          <strong>{datasetProgressPercent}%</strong>
                        </div>
                        <div className="progress-track">
                          <span style={{ width: `${datasetProgressPercent}%` }} />
                        </div>
                      </div>
                    )}
                    {dataset.length > 0 && (
                      <div className="generated-block">
                        <div className="generated-meta">
                            <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                                <span>
                                <strong>{dataset.length}</strong> зображень згенеровано
                                </span>
                                {generationDuration !== null && (
                                <span className="time-badge">
                                    Час: {formatDuration(generationDuration)}
                                </span>
                                )}
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                              {zipFile && (
                                <button
                                  className="text-button"
                                  onClick={downloadDatasetZip}
                                  title="Завантажити згенерований ZIP-архів"
                                >
                                  <Icon name="download" size={15} />
                                  Скачати ZIP
                                </button>
                              )}
                              <button
                                className="text-button danger"
                                onClick={() => {
                                  dataset.forEach((image) => {
                                    if (image.src.startsWith("blob:")) {
                                      URL.revokeObjectURL(image.src)
                                    }
                                  })
                                  zipValidationRef.current += 1
                                  setIsValidatingZip(false)
                                  setDataset([])
                                  setZipFile(null)
                                  setZipOrigin(null)
                                  setZipError("")
                                  setZipNotice("")
                                  setGenerationDuration(null)
                                }}
                              >
                                <Icon name="trash" size={15} />
                                Очистити
                              </button>
                            </div>
                        </div>
                        <div className="thumbnail-strip">
                          {dataset.map((image) => (
                            <button
                              key={image.id}
                              onClick={() => {
                                setSelectedDatasetImage(image)
                                setModal("dataset")
                              }}
                            >
                              <img alt={image.name} src={image.src} />
                              <span>{String(image.id + 1).padStart(2, "0")}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="dataset-area">
                    <FileDrop
                      accept=".zip,application/zip"
                      file={zipFile}
                      label="ZIP-архів із датасетом"
                      onChange={handleZipUpload}
                    />
                    {isValidatingZip && (
                      <p className="field-hint">Перевіряємо розміри зображень в архіві…</p>
                    )}
                    {zipError && <p className="field-error">{zipError}</p>}
                    {!zipError && zipNotice && (
                      <p className="field-success">{zipNotice}</p>
                    )}
                  </div>
                )}

                <div
                  className={`model-status ${
                    isTraining ? "training" : modelReady ? "ready" : ""
                  }`}
                >
                  <span className="model-status-dot" />
                  <span>Статус моделі</span>
                  <strong>
                    {isTraining
                      ? "Модель навчається"
                      : modelReady
                      ? "Модель готова до роботи"
                      : "Модель відсутня"}
                  </strong>
                </div>

                <div className="training-panel">
                  <div className="training-copy">
                    <h3>Навчання моделі</h3>
                    <p>{getTrainingHint()}</p>
                  </div>
                  <button
                    aria-label="Налаштування навчання"
                    className="icon-button light"
                    onClick={() => setModal("train-settings")}
                  >
                    <Icon name="settings" />
                  </button>
                  <button
                    className="primary-button light-button"
                    disabled={!trainingReady || isTraining}
                    onClick={startTraining}
                  >
                    <Icon name="play" />
                    {isTraining ? "Навчаємо..." : "Тренувати модель"}
                  </button>
                  <div className="progress-wrap">
                    <div className="progress-label">
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span>
                        {modelReady && !isTraining
                            ? "Модель навчено"
                            : isTraining
                            ? "Навчаємо модель"
                            : "Очікує запуску"}
                        </span>
                        {trainingDuration !== null && trainingProgress === 100 && (
                        <span className="time-badge light">
                            Час: {formatDuration(trainingDuration)}
                        </span>
                        )}
                    </div>
                    <strong>{displayedTrainingProgress}%</strong>
                    </div>
                    <div className="progress-track">
                      <span style={{ width: `${displayedTrainingProgress}%` }} />
                    </div>
                  </div>
                </div>
                {trainingError && (
                  <p className="field-error analysis-error">{trainingError}</p>
                )}

                <div className="model-controls">
                  <div className="model-controls-head">
                    <div>
                      <h3>Керування моделлю</h3>
                      <p>
                        Експортуйте навчену модель у ZIP-бандл або завантажте
                        сумісну модель.
                      </p>
                    </div>
                    <button
                      className="secondary-button"
                      disabled={!modelReady || isTraining || isDownloadingModel}
                      onClick={downloadModel}
                    >
                      {isDownloadingModel ? (
                        <span className="button-spinner" aria-hidden="true" />
                      ) : (
                        <Icon name="download" />
                      )}
                      {isDownloadingModel ? "Скачування..." : "Скачати модель"}
                    </button>
                  </div>

                  <div className="model-load-row">
                    <FileDrop
                      accept=".zip,application/zip"
                      file={modelFile}
                      label="ZIP-архів моделі"
                      onChange={(file) => {
                        setModelFile(file)
                        setModelError("")
                        setModelNotice("")
                      }}
                    />
                    <button
                      className="secondary-button"
                      disabled={!modelFile || isTraining || isLoadingModel}
                      onClick={loadModelFile}
                    >
                      {isLoadingModel ? (
                        <span className="button-spinner" aria-hidden="true" />
                      ) : (
                        <Icon name="upload" />
                      )}
                      {isLoadingModel ? "Завантаження..." : "Завантажити модель"}
                    </button>
                  </div>

                  {modelError && (
                    <p className="field-error model-control-message">{modelError}</p>
                  )}
                  {!modelError && modelNotice && (
                    <p className="field-success model-control-message">
                      {modelNotice}
                    </p>
                  )}
                  {/* {modelInfo && (
                    <p className="field-hint model-control-message">
                      {modelInfo.framework === "tensorflow"
                        ? "TensorFlow/Keras"
                        : modelInfo.framework || "Модель"}{" "}
                      · {modelInfo.architecture || "архітектура невідома"} ·
                      threshold {formatDecimal(modelInfo.threshold, 2)}
                    </p>
                  )} */}
                </div>
              </div>
          </section>

          <section className="zone processing-zone">
            <div className="zone-heading">
              <span className="zone-number">02</span>
              <div>
                <h2>Опрацювання</h2>
              </div>
            </div>

            <div className="processing-grid">
              <div className="visual-column">
                <div className="visual-label">
                  <span>Вхідне зображення</span>
                  <small>PNG</small>
                </div>
                <div className="image-stage">
                  {sourcePreview ? (
                    <img alt="Вхідне зображення" src={sourcePreview} />
                  ) : (
                    <label className="stage-placeholder" htmlFor="source-image">
                      <input
                        accept="image/png, .png"
                        id="source-image"
                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                          onSourceFile(event.target.files?.[0] ?? null)
                        }
                        type="file"
                      />
                      <span className="upload-icon">
                        <Icon name="upload" />
                      </span>
                      <strong>Завантажити PNG</strong>
                      <small>Оберіть зображення для аналізу</small>
                    </label>
                  )}
                </div>
                {sourceGridPreview && (
                  <button
                    className="replace-file preview-image"
                    onClick={() => setModal("source-grid")}
                  >
                    <Icon name="adjust" size={14} /> Показати зображення із сіткою
                  </button>
                )}
                <p className="field-hint">
                  Потрібне зображення {imageSize}×{imageSize} px.
                </p>
                {sourceError && <p className="field-error">{sourceError}</p>}
                {sourceFile && (
                  <button
                    className="replace-file"
                    onClick={() => onSourceFile(null)}
                  >
                    <Icon name="trash" size={14} /> Видалити зображення
                  </button>
                )}
              </div>

              <div className="visual-column">
                <div className="visual-label">
                  <span>Різницева сітка</span>
                  <small>Результат</small>
                </div>
                <div className="image-stage difference">
                  {differencePreview ? (
                    <img alt="Різницева сітка" src={differencePreview} />
                  ) : (
                    <div className="stage-placeholder muted">
                      <span className="grid-icon">
                        <Icon name="adjust" />
                      </span>
                      <strong>
                        {processing ? "Обробляємо..." : "Ще не оброблено"}
                      </strong>
                      <small>Сітка з’явиться після аналізу</small>
                    </div>
                  )}
                </div>
                {differencePreview && (
                  <>
                    <button
                      className="replace-file download"
                      onClick={downloadDifferenceGrid}
                    >
                      <Icon name="download" size={14} /> Завантажити зображення
                    </button>
                    <p className="field-hint">
                      Червоний — ділянки темніші за еталон, синій — світліші.
                    </p>
                  </>
                )}
              </div>
            </div>

            <button
              className="analyze-button"
              disabled={!sourceFile || processing || Boolean(sourceError) || !reference}
              onClick={processImage}
            >
              {processing ? (
                    <span className="button-spinner" aria-hidden="true" />
                ) : (
                    <Icon name="spark" />
                )}
                {processing
                  ? "Формуємо різницеву сітку..."
                  : !reference
                  ? "Спочатку згенеруйте еталон"
                  : sourceError
                  ? "Виправте розмір зображення"
                  : "Виявити дефекти"}
            </button>

            {analysisError && (
              <p className="field-error analysis-error">{analysisError}</p>
            )}

            <div
              className={`result-card ${
                prediction
                  ? prediction.has_defect
                    ? "defects"
                    : "clean"
                  : differencePreview
                  ? "clean"
                  : ""
              }`}
            >
              <div className="result-icon">
                <Icon
                  name={
                    prediction
                      ? prediction.has_defect
                        ? "adjust"
                        : "check"
                      : differencePreview
                      ? "check"
                      : processing
                      ? "adjust"
                      : "image"
                  }
                />
              </div>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className="eyebrow">Результат обробки</span>
                </div>
                <h3>
                  {prediction
                    ? prediction.has_defect
                      ? "Модель виявила дефект"
                      : "Модель не виявила дефектів"
                    : differencePreview
                    ? "Різницеву сітку сформовано"
                    : processing
                    ? "Формуємо різницеву сітку"
                    : "Очікуємо на аналіз"}
                </h3>
                <p>
                  {prediction
                    ? `Ймовірність дефекту: ${(prediction.probability * 100).toFixed(
                        1,
                      )}%. Впевненість моделі: ${(prediction.confidence * 100).toFixed(1)}%.`
                    : differencePreview
                    ? predictionMessage ||
                      "Вхідне зображення порівняно з еталоном. Для класифікації дефектів навчіть модель."
                    : !reference
                    ? "Спочатку згенеруйте еталонне зображення."
                    : sourceError
                    ? "Виправте розмір вхідного зображення, щоб продовжити."
                    : !sourceFile
                    ? "Завантажте зображення та запустіть перевірку."
                    : "Натисніть «Виявити дефекти», щоб сформувати різницеву сітку."}
                </p>
              </div>
              <div className="result-actions">
                {differencePreview && (
                  <span className="result-badge">
                    {prediction
                      ? prediction.has_defect
                        ? "ДЕФЕКТ"
                        : "БЕЗ ДЕФЕКТУ"
                      : "ГОТОВО"}
                  </span>
                )}
                {modelReady && (
                  <button
                    className="text-button details-button"
                    onClick={() => setModal("model-details")}
                  >
                    <Icon name="adjust" size={14} /> Детальніше
                  </button>
                )}
              </div>
            </div>

            <div className="logs">
              <div className="logs-header">
                <div>
                  <h3>Логи системи</h3>
                </div>
                <button className="text-button" onClick={() => setLogs([])}>
                  Очистити
                </button>
              </div>
              <div className="log-list" ref={logRef}>
                {logs.length ? (
                  logs.map((log, index) => (
                    <div className="log-row" key={`${log.timestamp}-${index}`}>
                      <span className={`log-dot ${getLogState(log.type)}`} />
                      <time>{formatLogTime(log.timestamp)}</time>
                      <span>{log.text}</span>
                    </div>
                  ))
                ) : (
                  <div className="empty-logs">Повідомлень ще немає</div>
                )}
              </div>
            </div>
          </section>
        </div>
      </main>

      {modal === "reference-settings" && (
        <Modal onClose={() => setModal(null)} title="Налаштування">
          <div className="modal-fields">
            <label>
              <span>Зерно рандому</span>
              <input
                onChange={(event) =>
                  setSeed(
                    event.target.value === "" ? null : Number(event.target.value),
                  )
                }
                type="number"
                value={seed ?? ""}
              />
              <small>Для відтворюваної генерації</small>
            </label>
            <label>
              <span>Розмір сторони</span>
              <select
                onChange={(event) => setImageSize(Number(event.target.value))}
                value={imageSize}
              >
                <option value="200">200 px</option>
                <option value="500">500 px</option>
                <option value="1000">1000 px</option>
              </select>
            </label>
            <label>
              <span>Розмір сітки grid</span>
              <select
                onChange={(event) => setGridSize(Number(event.target.value))}
                value={gridSize}
              >
                <option value="10">10 × 10</option>
                <option value="20">20 × 20</option>
              </select>
            </label>
          </div>
          <div className="modal-actions">
            <button className="primary-button" onClick={saveSettings}>
              Зберегти налаштування
            </button>
          </div>
        </Modal>
      )}

      {modal === "train-settings" && (
        <Modal onClose={() => setModal(null)} title="Налаштування навчання">
          <div className="modal-fields">
            <label>
              <span>Кількість епох</span>
              <input
                min="1"
                onChange={(event) => setEpochs(Number(event.target.value))}
                type="number"
                value={epochs}
              />
            </label>
            <label>
              <span>Швидкість навчання</span>
              <input
                min="0.001"
                onChange={(event) => setLearningRate(Number(event.target.value))}
                step="0.001"
                type="number"
                value={learningRate}
              />
            </label>
            <label>
              <span>Кількість прихованих шарів</span>
              <select
                onChange={(event) =>
                  setHiddenLayers(Number(event.target.value) as 0 | 1)
                }
                value={hiddenLayers}
              >
                <option value="0">0 — лінійна модель</option>
                <option value="1">1 — прихований шар</option>
              </select>
            </label>
            {hiddenLayers === 1 && (
              <label>
                <span>Нейронів у прихованому шарі</span>
                <input
                  max="256"
                  min="1"
                  onChange={(event) => setHiddenUnits(Number(event.target.value))}
                  type="number"
                  value={hiddenUnits}
                />
                <small>Від 1 до 256 нейронів, активація sigmoid</small>
              </label>
            )}
            <p className="field-hint">
              {hiddenLayers === 1
                ? `Input → Normalization → Dense(${hiddenUnits}, relu) → Dropout() → Dense(1, sigmoid)`
                : "Input → Normalization → Dense(1, sigmoid)"}
            </p>
          </div>
          <div className="modal-actions">
            <button className="primary-button" onClick={saveTrainingSettings}>
              Зберегти налаштування
            </button>
          </div>
        </Modal>
      )}

      {modal === "model-details" && modelInfo && (
        <Modal onClose={() => setModal(null)} title="Деталі моделі">
          <div className="model-details">
            <section className="details-section">
              <div className="details-section-head">
                <h3>Якість на тестовій вибірці</h3>
                <span
                  className={`compatibility-badge ${
                    modelCompatible ? "ok" : "bad"
                  }`}
                >
                  {modelCompatible ? "Сумісна" : "Несумісна"}
                </span>
              </div>
              <div className="metric-grid">
                <div className="metric-card">
                  <span>Test accuracy</span>
                  <strong>{formatPercent(modelMetrics?.test_accuracy)}</strong>
                </div>
                <div className="metric-card">
                  <span>Balanced accuracy</span>
                  <strong>
                    {formatPercent(modelMetrics?.test_balanced_accuracy)}
                  </strong>
                </div>
                <div className="metric-card">
                  <span>Precision</span>
                  <strong>{formatPercent(modelMetrics?.test_precision)}</strong>
                </div>
                <div className="metric-card">
                  <span>Recall</span>
                  <strong>{formatPercent(modelMetrics?.test_recall)}</strong>
                </div>
                <div className="metric-card">
                  <span>F1</span>
                  <strong>{formatPercent(modelMetrics?.test_f1)}</strong>
                </div>
                <div className="metric-card">
                  <span>ROC AUC</span>
                  <strong>{formatDecimal(modelMetrics?.test_roc_auc, 4)}</strong>
                </div>
              </div>
            </section>

            <section className="details-section">
              <h3>Confusion matrix</h3>
              {modelMetrics?.test_confusion_matrix ? (
                <table className="confusion-matrix">
                  <thead>
                    <tr>
                      <th />
                      <th>Прогноз class0</th>
                      <th>Прогноз class1</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Факт class0</th>
                      <td>{modelMetrics.test_confusion_matrix[0]?.[0] ?? "—"}</td>
                      <td>{modelMetrics.test_confusion_matrix[0]?.[1] ?? "—"}</td>
                    </tr>
                    <tr>
                      <th>Факт class1</th>
                      <td>{modelMetrics.test_confusion_matrix[1]?.[0] ?? "—"}</td>
                      <td>{modelMetrics.test_confusion_matrix[1]?.[1] ?? "—"}</td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <p className="field-hint">Матриця помилок недоступна.</p>
              )}
            </section>

            <section className="details-section">
              <h3>Метрики за класами</h3>
              {modelMetrics?.per_class ? (
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Клас</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>class0</td>
                      <td>{formatPercent(modelMetrics.per_class.class0.precision)}</td>
                      <td>{formatPercent(modelMetrics.per_class.class0.recall)}</td>
                      <td>{formatPercent(modelMetrics.per_class.class0.f1)}</td>
                      <td>{modelMetrics.per_class.class0.support}</td>
                    </tr>
                    <tr>
                      <td>class1</td>
                      <td>{formatPercent(modelMetrics.per_class.class1.precision)}</td>
                      <td>{formatPercent(modelMetrics.per_class.class1.recall)}</td>
                      <td>{formatPercent(modelMetrics.per_class.class1.f1)}</td>
                      <td>{modelMetrics.per_class.class1.support}</td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <p className="field-hint">Метрики за класами недоступні.</p>
              )}
            </section>

            <section className="details-section">
              <h3>Дані та навчання</h3>
              <div className="details-grid">
                <div>
                  <span>Train / Test</span>
                  <strong>
                    {modelMetrics?.train_samples ?? "—"} /{" "}
                    {modelMetrics?.test_samples ?? "—"}
                  </strong>
                </div>
                <div>
                  <span>Train / Test loss</span>
                  <strong>
                    {formatDecimal(modelMetrics?.train_loss)} /{" "}
                    {formatDecimal(modelMetrics?.test_loss)}
                  </strong>
                </div>
                <div>
                  <span>Розподіл класів</span>
                  <strong>
                    class0 {modelMetrics?.class_distribution?.class0 ?? "—"} ·
                    class1 {modelMetrics?.class_distribution?.class1 ?? "—"}
                  </strong>
                </div>
                <div>
                  <span>Class weights</span>
                  <strong>
                    class0 {formatDecimal(modelMetrics?.class_weights?.class0)} ·
                    class1 {formatDecimal(modelMetrics?.class_weights?.class1)}
                  </strong>
                </div>
                <div>
                  <span>Епохи</span>
                  <strong>{modelInfo.epochs}</strong>
                </div>
                <div>
                  <span>Learning rate</span>
                  <strong>{modelInfo.learning_rate}</strong>
                </div>
                <div>
                  <span>Приховані шари</span>
                  <strong>{modelInfo.hidden_layers ?? "—"}</strong>
                </div>
                <div>
                  <span>Нейрони</span>
                  <strong>{modelInfo.hidden_units ?? "—"}</strong>
                </div>
                <div>
                  <span>Seed</span>
                  <strong>{modelInfo.random_state ?? "—"}</strong>
                </div>
                <div>
                  <span>Threshold</span>
                  <strong>{formatDecimal(modelInfo.threshold, 2)}</strong>
                </div>
                <div>
                  <span>Навчено</span>
                  <strong>{formatDateTime(modelInfo.trained_at)}</strong>
                </div>
                <div>
                  <span>Тривалість</span>
                  <strong>
                    {formatDecimal(
                      modelMetrics?.duration_seconds ??
                        modelInfo.duration_seconds,
                      2,
                    )}{" "}
                    с
                  </strong>
                </div>
                <div>
                  <span>Джерело</span>
                  <strong>{formatModelSource(modelInfo.source)}</strong>
                </div>
                <div>
                  <span>Архітектура</span>
                  <strong>{modelInfo.architecture || "—"}</strong>
                </div>
              </div>
            </section>

            <section className="details-section">
              <h3>Сумісність із поточними налаштуваннями</h3>
              <div
                className={`compatibility ${
                  modelCompatible ? "ok" : "bad"
                }`}
              >
                <strong>
                  {modelCompatible
                    ? "Модель сумісна з поточними налаштуваннями."
                    : "Модель несумісна з поточними налаштуваннями."}
                </strong>
                <table className="metrics-table compatibility-table">
                  <thead>
                    <tr>
                      <th>Параметр</th>
                      <th>Модель</th>
                      <th>Поточні</th>
                      <th>Статус</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>img_size</td>
                      <td>{modelInfo.img_size}</td>
                      <td>{imageSize}</td>
                      <td>{modelInfo.img_size === imageSize ? "OK" : "Різниця"}</td>
                    </tr>
                    <tr>
                      <td>grid_cells</td>
                      <td>{modelInfo.grid_cells}</td>
                      <td>{gridSize}</td>
                      <td>
                        {modelInfo.grid_cells === gridSize ? "OK" : "Різниця"}
                      </td>
                    </tr>
                    <tr>
                      <td>feature_count</td>
                      <td>{modelInfo.feature_count}</td>
                      <td>{currentFeatureCount}</td>
                      <td>
                        {modelInfo.feature_count === currentFeatureCount
                          ? "OK"
                          : "Різниця"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </Modal>
      )}

      {modal === "reference" && reference && (
        <Modal onClose={() => setModal(null)} title="Еталонне зображення">
          <div className="preview-modal">
            <img alt="Еталонне зображення" src={reference} />
          </div>
          <div className="modal-actions spread">
            <button
              className="primary-button"
              onClick={() => downloadData(reference, "reference.png")}
            >
              <Icon name="download" /> Скачати PNG
            </button>
          </div>
        </Modal>
      )}

      {modal === "source-grid" && sourceGridPreview && (
        <Modal onClose={() => setModal(null)} title="Вхідне зображення із сіткою">
          <div className="preview-modal">
            <img
              alt="Вхідне зображення із сіткою"
              src={sourceGridPreview}
            />
          </div>
          <div className="modal-actions spread">
            <span>Сітка {gridSize}×{gridSize}</span>
            <button
              className="primary-button"
              onClick={() =>
                downloadData(sourceGridPreview, "source_with_grid.png")
              }
            >
              <Icon name="download" /> Скачати PNG
            </button>
          </div>
        </Modal>
      )}

      {modal === "dataset" && selectedDatasetImage && (
        <Modal onClose={() => setModal(null)} title={selectedDatasetImage.name}>
          <div className="preview-modal">
            <img alt={selectedDatasetImage.name} src={selectedDatasetImage.src} />
          </div>
          <div className="modal-actions spread">
            <span>Зображення з датасету</span>
            <button
              className="primary-button"
              onClick={() =>
                downloadData(selectedDatasetImage.src, selectedDatasetImage.name)
              }
            >
              <Icon name="download" /> Скачати PNG
            </button>
          </div>
        </Modal>
      )}
    </div>
  )
}