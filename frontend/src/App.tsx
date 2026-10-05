import { ChangeEvent, ReactNode, useEffect, useRef, useState } from "react"
import JSZip from "jszip"

type IconName = "adjust" | "archive" | "check" | "close" | "download" | "image" | "play" | "settings" | "spark" | "trash" | "upload"

type GeneratedImage = {
  id: number
  name: string
  src: string
}

const API_BASE_URL = "http://127.0.0.1:8000"

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
  const [mainTab, setMainTab] = useState<"train" | "model">("train")
  const [datasetTab, setDatasetTab] = useState<"generate" | "upload">("generate")
  const [modal, setModal] = useState<"reference-settings" | "train-settings" | "reference" | "dataset" | null>(null)
  
  const [seed, setSeed] = useState(42)
  const [imageSize, setImageSize] = useState(200)
  const [gridSize, setGridSize] = useState(10)
  const [reference, setReference] = useState("")
  const [datasetCount, setDatasetCount] = useState(250)
  const [dataset, setDataset] = useState<GeneratedImage[]>([])
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false)
  const [selectedDatasetImage, setSelectedDatasetImage] = useState<GeneratedImage | null>(null)
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [epochs, setEpochs] = useState(200)
  const [learningRate, setLearningRate] = useState(0.1)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [isTraining, setIsTraining] = useState(false)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [sourcePreview, setSourcePreview] = useState("")
  const [differencePreview, setDifferencePreview] = useState("")
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<"defects" | "clean" | null>(null)
  const [logs, setLogs] = useState<{ time: string; text: string; state: string }[]>([])
  const logRef = useRef<HTMLDivElement>(null)
  const [isGeneratingReference, setIsGeneratingReference] = useState(false)
  const [generationDuration, setGenerationDuration] = useState<number | null>(null)
  const [trainingDuration, setTrainingDuration] = useState<number | null>(null)

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${Math.round(ms)} мс`
    return `${(ms / 1000).toFixed(2)} с`
    }

  useEffect(() => {
    return () => {
      if (sourcePreview.startsWith("blob:")) URL.revokeObjectURL(sourcePreview)
      if (reference.startsWith("blob:")) URL.revokeObjectURL(reference)
      dataset.forEach((img) => {
        if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src)
      })
    }
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [logs])

  const addLog = (text: string, state = "idle") => {
    setLogs((current) => [
      ...current,
      {
        time: new Date().toLocaleTimeString("uk-UA", { hour12: false }),
        text,
        state,
      },
    ])
  }

  // --- 1. ВСТАНОВЛЕННЯ НАЛАШТУВАНЬ НА БЕКЕНДІ ---
  const saveSettings = async () => {
    try {
      addLog("Оновлення налаштувань системи...")
      const response = await fetch(`${API_BASE_URL}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          random_seed: seed,
          img_size: imageSize,
          grid_cells: gridSize,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Помилка збереження налаштувань")
      }

      addLog("Налаштування успішно збережено на бекенді", "ok")
      setModal(null)
    } catch (err: any) {
      addLog(`Помилка: ${err.message}`, "warn")
    }
  }

  // --- 2. ГЕНЕРАЦІЯ ТА ОТРИМАННЯ ЕТАЛОНА З БЕКЕНДУ ---
    const generateReference = async () => {
    if (isGeneratingReference) return
    setIsGeneratingReference(true)
    try {
        addLog("Запит еталонного зображення...")
        const response = await fetch(`${API_BASE_URL}/template?t=${Date.now()}`)
        if (!response.ok) throw new Error("Не вдалося отримати еталон")

        const blob = await response.blob()
        if (reference.startsWith("blob:")) URL.revokeObjectURL(reference)
        const url = URL.createObjectURL(blob)
        setReference(url)
        addLog("Еталонне зображення отримано від сервера", "ok")
    } catch (err: any) {
        addLog(`Помилка генерації еталона: ${err.message}`, "warn")
    } finally {
        setIsGeneratingReference(false)
    }
    }

  // --- 3. ГЕНЕРАЦІЯ ДАТАСЕТУ (ZIP З ФАЙЛАМИ) ТА ВІДОБРАЖЕННЯ НА ФРОНТІ ---
  const generateDataset = async () => {
    if (isGeneratingDataset) return
    setIsGeneratingDataset(true)
    setGenerationDuration(null)
    const startTime = performance.now()
    const count = Math.max(1, datasetCount || 1)
    addLog(`Запит на генерацію датасету: ${count} пар зображень...`)

    try {
      const response = await fetch(`${API_BASE_URL}/dataset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ total_samples: count }),
      })

      if (!response.ok) throw new Error("Помилка генерації на бекенді")

      const zipBlob = await response.blob()
      addLog("ZIP-архів отримано, розархівовуємо для відображення...", "idle")

      // Розархівовуємо ZIP в оперативній пам'яті браузера
      const zip = await JSZip.loadAsync(zipBlob)
      const images: GeneratedImage[] = []
      let id = 0

      // Зберігаємо сам отриманий zip файл для подальшого тренування
      setZipFile(new File([zipBlob], "dataset.zip", { type: "application/zip" }))

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

      setDataset(images)
      const duration = performance.now() - startTime
      setGenerationDuration(duration)
      addLog(`Датасет згенеровано: ${images.length} файлів`, "ok")
    } catch (err: any) {
      addLog(`Помилка генерації датасету: ${err.message}`, "warn")
    } finally {
      setIsGeneratingDataset(false)
    }
  }

  const startTraining = () => {
    if ((!dataset.length && !zipFile) || isTraining) return
    setIsTraining(true)
    setTrainingProgress(4)
    setTrainingDuration(null)
    const startTime = performance.now()
    addLog(`Навчання моделі розпочато · ${epochs} епох`)
    const interval = window.setInterval(() => {
      setTrainingProgress((current) => {
        const next = Math.min(current + 4, 100)
        if (next === 100) {
          window.clearInterval(interval)
          setIsTraining(false)
          const duration = performance.now() - startTime
          setTrainingDuration(duration)
          addLog("Навчання моделі завершено", "ok")
        }
        return next
      })
    }, 90)
  }

  const onSourceFile = (file: File | null) => {
    setSourceFile(file)
    setDifferencePreview("")
    setResult(null)
    if (sourcePreview.startsWith("blob:")) URL.revokeObjectURL(sourcePreview)
    setSourcePreview(file ? URL.createObjectURL(file) : "")
    if (file) addLog(`Зображення «${file.name}» завантажено`, "ok")
  }

  const processImage = () => {
    if (!sourceFile || processing) return
    setProcessing(true)
    setResult(null)
    addLog("Порівнюємо геометрію з еталоном...")
    
    // Заглушка обробки інференсу до створення відповідного ендпоінта
    window.setTimeout(() => {
      const hasDefects = sourceFile.name.includes("class1")
      setResult(hasDefects ? "defects" : "clean")
      setProcessing(false)
      addLog(
        hasDefects
          ? "Аналіз завершено: знайдено дефекти"
          : "Аналіз завершено: дефектів немає",
        hasDefects ? "warn" : "ok",
      )
    }, 800)
  }

  const trainingReady = dataset.length > 0 || Boolean(zipFile)

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

            <div className="main-tabs" role="tablist">
              <button
                className={mainTab === "train" ? "active" : ""}
                onClick={() => setMainTab("train")}
                role="tab"
              >
                Тренування моделі
              </button>
              {/* <button
                className={mainTab === "model" ? "active" : ""}
                onClick={() => setMainTab("model")}
                role="tab"
              >
                Завантажити модель
              </button> */}
            </div>

            {mainTab === "train" ? (
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
                          min="1"
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
                        {isGeneratingDataset ? (
                            <span className="button-spinner" aria-hidden="true" />
                        ) : (
                            <Icon name="spark" />
                        )}
                        {isGeneratingDataset ? "Генерація..." : "Генерувати"}
                      </button>
                    </div>
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
                            <button
                                className="text-button danger"
                                onClick={() => {
                                setDataset([])
                                setGenerationDuration(null)
                                }}
                            >
                                <Icon name="trash" size={15} />
                                Очистити
                            </button>
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
                      onChange={(file) => {
                        setZipFile(file)
                        if (file) addLog(`Архів «${file.name}» завантажено`, "ok")
                      }}
                    />
                  </div>
                )}

                <div
                  className={`model-status ${
                    trainingProgress === 100 || modelFile ? "ready" : ""
                  }`}
                >
                  <span className="model-status-dot" />
                  <span>Статус моделі</span>
                  <strong>
                    {trainingProgress === 100 || modelFile
                      ? "Модель готова до роботи"
                      : "Модель відсутня"}
                  </strong>
                </div>

                <div className="training-panel">
                  <div className="training-copy">
                    <h3>Навчання моделі</h3>
                    <p>
                      {trainingReady
                        ? `Дані готові · ${epochs} епох · швидкість ${learningRate}`
                        : "Спочатку згенеруйте або завантажте датасет"}
                    </p>
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
                        {trainingProgress === 100
                            ? "Навчання завершено"
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
                    <strong>{trainingProgress}%</strong>
                    </div>
                    <div className="progress-track">
                      <span style={{ width: `${trainingProgress}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="tab-content model-upload">
                <FileDrop
                  accept=".onnx,.h5,.pt,.pth,.model"
                  file={modelFile}
                  label="Файл навченої моделі"
                  onChange={setModelFile}
                />
                <button
                  className="secondary-button"
                  disabled={!modelFile}
                  onClick={() => {
                    if (modelFile)
                      downloadData(
                        URL.createObjectURL(modelFile),
                        modelFile.name,
                      )
                  }}
                >
                  <Icon name="download" />
                  Скачати модель
                </button>
              </div>
            )}
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
                        accept="image/png, image/jpeg, image/bmp, .png, .jpg, .jpeg, .bmp"
                        id="source-image"
                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                          onSourceFile(event.target.files?.[0] ?? null)
                        }
                        type="file"
                      />
                      <span className="upload-icon">
                        <Icon name="upload" />
                      </span>
                      <strong>Завантажити PNG, BMP, JPG</strong>
                      <small>Оберіть зображення для аналізу</small>
                    </label>
                  )}
                </div>
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
              </div>
            </div>

            <button
              className="analyze-button"
              disabled={!sourceFile || processing}
              onClick={processImage}
            >
              {processing ? (
                    <span className="button-spinner" aria-hidden="true" />
                ) : (
                    <Icon name="spark" />
                )}
                {processing ? "Виконуємо аналіз..." : "Виявити дефекти"}
            </button>

            <div className={`result-card ${result ?? ""}`}>
              <div className="result-icon">
                <Icon
                  name={
                    result === "clean"
                      ? "check"
                      : result === "defects"
                      ? "adjust"
                      : "image"
                  }
                />
              </div>
              <div>
                <span className="eyebrow">Результат обробки</span>
                <h3>
                  {result === "clean"
                    ? "Дефектів не виявлено"
                    : result === "defects"
                    ? "Виявлено дефекти"
                    : "Очікуємо на аналіз"}
                </h3>
                <p>
                  {result
                    ? result === "clean"
                      ? "Геометрія відповідає еталонному зображенню."
                      : "На різницевій сітці позначено підозрілі ділянки."
                    : "Завантажте PNG-зображення та запустіть перевірку."}
                </p>
              </div>
              {result && (
                <span className="result-badge">
                  {result === "clean" ? "OK" : "УВАГА"}
                </span>
              )}
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
                    <div className="log-row" key={`${log.time}-${index}`}>
                      <span className={`log-dot ${log.state}`} />
                      <time>{log.time}</time>
                      <span>{log.text}</span>
                    </div>
                  ))
                ) : (
                  <div className="empty-logs">Логи очищено</div>
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
                onChange={(event) => setSeed(Number(event.target.value))}
                type="number"
                value={seed}
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
          </div>
          <div className="modal-actions">
            <button className="primary-button" onClick={() => setModal(null)}>
              Зберегти налаштування
            </button>
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