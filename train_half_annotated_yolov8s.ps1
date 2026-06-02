$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunName = "yolov8s_640_e50"
$Project = Join-Path $Root "runs\half-annotated"
$RunDir = Join-Path $Project $RunName
$ModelOut = Join-Path $Root "models\half-annotated\yolov8s_640_e50_best.pt"
$DataYaml = "C:\Users\Vuk\Desktop\Master rad\Datasets\Master Half Annotated.yolov8\data.yaml"
$TimeLog = Join-Path $RunDir "training_time.txt"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$sw = [System.Diagnostics.Stopwatch]::StartNew()

try {
    & (Join-Path $Root "venv\Scripts\yolo.exe") detect train `
        model=yolov8s.pt `
        data="$DataYaml" `
        epochs=50 `
        imgsz=640 `
        batch=8 `
        device=0 `
        workers=0 `
        patience=20 `
        cache=False `
        project="$Project" `
        name="$RunName" `
        exist_ok=True `
        plots=True

    if ($LASTEXITCODE -ne 0) {
        throw "YOLO training failed. Exit code: $LASTEXITCODE"
    }

    Copy-Item `
        -LiteralPath (Join-Path $RunDir "weights\best.pt") `
        -Destination $ModelOut `
        -Force

    "Best model copied to: $ModelOut"
}
finally {
    $sw.Stop()

    $timeText = "Training time: {0:hh\:mm\:ss}" -f $sw.Elapsed
    $timeText | Tee-Object -FilePath $TimeLog
    "Time log saved to: $TimeLog"
}
