# Sets up and starts DALEEL on Windows; start.bat in the project folder runs this script.
# The first run downloads Python 3.10.11, the packages in requirements.txt and the two model files
# into this folder (.tools, .venv and weights). Later runs start DALEEL straight away.
# Nothing is installed elsewhere on the computer, and the instance folder (accounts, passwords,
# missions and evidence) is never changed by the setup.
param([switch]$SetupOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # a progress bar makes Windows PowerShell downloads very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root '.tools'
$Venv = Join-Path $Root '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Marker = Join-Path $Venv 'daleel-setup.txt'
$Database = Join-Path $Root 'instance\v2\daleel.db'

# DALEEL was developed, tested and evaluated with Python 3.10.11 from python.org, pip 23.0.1 and the
# package versions in requirements-lock.txt; PyTorch 2.3.1 does not install on Python 3.13 or later.
# Python comes from python.org's official NuGet package (published by the Python Software Foundation),
# pinned by its SHA-256; its python.exe must also carry a valid Python Software Foundation signature.
$PythonVersion = '3.10.11'
$PythonUrl = "https://api.nuget.org/v3-flatcontainer/python/$PythonVersion/python.$PythonVersion.nupkg"
$PythonSha256 = '7C6F99B160A36A7E09492DFCFF2B0A3A60BB5229CA44CDCC3ECB32871A6144D0'
$PythonDir = Join-Path $Tools "python-$PythonVersion"
$BasePython = Join-Path $PythonDir 'python.exe'

$env:PIP_CACHE_DIR = Join-Path $Tools 'pip-cache'  # finished downloads are kept if the setup is interrupted
$env:PIP_CONFIG_FILE = 'nul'                       # os.devnull: ignore pip settings made for other projects
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
foreach ($name in 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV') {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

function Write-Step($text) {
    Write-Host ''
    Write-Host $text -ForegroundColor Cyan
}

function Invoke-Tool([string]$failure, [scriptblock]$command) {
    # Native programs report progress on stderr, which must not count as a PowerShell error.
    $ErrorActionPreference = 'Continue'
    & $command
    if ($LASTEXITCODE -ne 0) { throw $failure }
}

function Get-SetupId {
    $files = 'requirements.txt', 'requirements-lock.txt' | ForEach-Object {
        (Get-FileHash (Join-Path $Root $_) -Algorithm SHA256).Hash
    }
    "python $PythonVersion $($files -join ' ')"
}

function Test-Setup {
    (Test-Path $Python) -and (Test-Path $Marker) -and ((Get-Content $Marker -Raw).Trim() -eq (Get-SetupId))
}

function Install-Python {
    if (Test-Path $BasePython) { return }
    Write-Host "Downloading Python $PythonVersion (16 MB) from python.org's NuGet package at nuget.org"
    $package = Join-Path $Tools "python.$PythonVersion.nupkg"
    try {
        Invoke-WebRequest $PythonUrl -OutFile $package -UseBasicParsing
    } catch {
        throw "Python $PythonVersion could not be downloaded. Check the internet connection."
    }
    if ((Get-FileHash $package -Algorithm SHA256).Hash -ne $PythonSha256) {
        Remove-Item $package
        throw "The downloaded Python $PythonVersion failed its SHA-256 check and was deleted."
    }
    $partial = "$PythonDir.part"
    if (Test-Path $partial) { Remove-Item -Recurse -Force $partial }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($package, $partial)
    $signature = Get-AuthenticodeSignature (Join-Path $partial 'tools\python.exe')
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Python Software Foundation') {
        Remove-Item -Recurse -Force $partial
        throw "The downloaded python.exe does not carry a valid Python Software Foundation signature."
    }
    Move-Item (Join-Path $partial 'tools') $PythonDir
    Remove-Item -Recurse -Force $partial
    Remove-Item $package
}

function Test-VenvPython {
    # The environment must be built on the Python checked above, not on another Python 3.10.
    if (-not (Test-Path $Python)) { return $false }
    $ErrorActionPreference = 'Continue'
    $same = & $Python -c "import os, sys; print(os.path.samefile(sys.base_prefix, sys.argv[1]))" $PythonDir 2>$null
    ($LASTEXITCODE -eq 0) -and ("$same".Trim() -eq 'True')
}

function Install-Daleel {
    Write-Step "[1/3] Getting Python $PythonVersion (first start only)"
    New-Item -ItemType Directory -Force $Tools | Out-Null
    Install-Python
    if ((Test-Path $Venv) -and -not (Test-VenvPython)) {
        if ((Get-Item $Venv -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'The .venv folder is a link to another folder. Remove the link, then run start.bat again.'
        }
        Write-Host "Replacing the existing .venv folder, which was not made with DALEEL's Python $PythonVersion."
        Write-Host 'Accounts, passwords, missions and evidence are kept in the instance folder and are not affected.'
        Remove-Item -Recurse -Force $Venv
    }
    if (-not (Test-Path $Python)) {
        Invoke-Tool "The Python $PythonVersion environment could not be created." { & $BasePython -m venv $Venv }
    }
    Write-Step '[2/3] Downloading and installing the Python packages (first start only; about 540 MB)'
    Write-Host 'The time this takes depends mainly on the speed of the internet connection.'
    Invoke-Tool 'The Python packages could not be installed.' {
        & $Python -m pip install --timeout 300 --retries 10 --no-warn-script-location `
            -r (Join-Path $Root 'requirements.txt') --constraint (Join-Path $Root 'requirements-lock.txt')
    }
    Set-Content $Marker (Get-SetupId) -Encoding ascii
    Write-Step '[3/3] Downloading and checking the two detection models (29 MB)'
}

function Test-Daleel([string]$url) {
    try {
        $page = Invoke-WebRequest "$url/login" -UseBasicParsing -TimeoutSec 3
        return $page.Content -match 'DALEEL'
    } catch {
        return $false
    }
}

function Test-PortInUse([int]$port) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $client.Connect('127.0.0.1', $port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Start-Daleel {
    Set-Location $Root
    if (-not (Test-Setup)) { Install-Daleel } else { Write-Host 'Checking the detection models...' }
    Invoke-Tool 'The detection models could not be downloaded or checked.' {
        & $Python (Join-Path $Root 'scripts\setup_weights.py') | Out-Null
    }
    if ($SetupOnly) {
        Write-Host ''
        Write-Host 'DALEEL is set up. Run start.bat to start it.' -ForegroundColor Green
        return
    }

    $port = '5000'
    if ($env:DALEEL_PORT) { $port = $env:DALEEL_PORT.Trim() }
    $url = "http://127.0.0.1:$port"
    if (Test-Daleel $url) {
        Write-Host "DALEEL is already running; opening $url"
        Start-Process $url
        return
    }
    if (($port -match '^\d+$') -and (Test-PortInUse ([int]$port))) {
        throw ("Another program is using port $port. Close it, or start DALEEL on another port by running " +
               "`$env:DALEEL_PORT = '5050'; .\start.bat in PowerShell.")
    }

    Write-Host ''
    if (-not (Test-Path $Database)) {
        Write-Host 'First start: DALEEL now creates the Administrator account and shows its temporary username' -ForegroundColor Yellow
        Write-Host 'and password below, once. To copy them, select them with the mouse and right-click.' -ForegroundColor Yellow
        Write-Host "(Ctrl+C without a selection stops DALEEL.) Then open $url in your browser." -ForegroundColor Yellow
    } else {
        Write-Host "Starting DALEEL; your browser opens at $url when it is ready."
        Start-Job -ArgumentList $url -ScriptBlock {
            param($url)
            for ($i = 0; $i -lt 180; $i++) {
                try {
                    Invoke-WebRequest "$url/login" -UseBasicParsing -TimeoutSec 2 | Out-Null
                    Start-Process $url
                    return
                } catch {
                    Start-Sleep -Seconds 1
                }
            }
        } | Out-Null
    }
    Write-Host 'Keep this window open while you use DALEEL. To stop DALEEL, close this window.'
    Write-Host ''
    $ErrorActionPreference = 'Continue'
    & $Python (Join-Path $Root 'run.py')
    exit $LASTEXITCODE
}

try {
    Start-Daleel
} catch {
    Write-Host ''
    Write-Host "DALEEL could not start: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Run start.bat again after fixing the problem; finished downloads are kept.'
    exit 1
}
