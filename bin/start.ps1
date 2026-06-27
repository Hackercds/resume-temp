# 简历 RAG 智能问答系统 - Windows 一键启动脚本
# 用法: .\bin\start.ps1 [-EsHost http://localhost:9200] [-Port 8080]

param(
    [string]$EsHost = "http://localhost:9200",
    [string]$Port = "8080"
)

$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $PSScriptRoot
Set-Location $AppDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  简历 RAG 智能问答系统 v1.1.0"
Write-Host "==========================================" -ForegroundColor Cyan

# 检查 Python
$PythonBin = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonBin) {
    Write-Host "❌ Python 未安装" -ForegroundColor Red
    exit 1
}
$PyVersion = python -c 'import sys; print("%d.%d" % sys.version_info[:2])'
Write-Host "✅ Python $PyVersion" -ForegroundColor Green

# 创建必要目录
New-Item -ItemType Directory -Force -Path log, models, uploads | Out-Null

# 检查依赖
$depCheck = python -c "import fastapi, elasticsearch, sentence_transformers" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  依赖未安装，正在执行 pip install -r requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
}
Write-Host "✅ 依赖已就绪" -ForegroundColor Green

# 检查 ES 连接
Write-Host "检查 ES 连接: $EsHost"
try {
    $null = Invoke-WebRequest -Uri $EsHost -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ ES 已连接" -ForegroundColor Green
} catch {
    Write-Host "⚠️  ES 未连接 ($EsHost)" -ForegroundColor Yellow
    Write-Host "   提示: 可通过 -EsHost 参数指定" -ForegroundColor Yellow
}

# 启动服务
$env:ES_HOST = $EsHost
$env:APP_PORT = $Port
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "监听端口: $Port"
Write-Host "前端入口: frontend\index.html"
Write-Host "API 文档: http://localhost:$Port/docs"
Write-Host "==========================================" -ForegroundColor Cyan

python main.py
