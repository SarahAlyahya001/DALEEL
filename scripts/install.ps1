# Installs DALEEL release v2.0.2 with one command on Windows (README, step 1):
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/SarahAlyahya001/DALEEL/v2.0.2/scripts/install.ps1 | iex"
# It downloads that release into the DALEEL folder of the user's home folder (or into DALEEL_DIR, if set)
# and runs start.bat there, which sets DALEEL up on the first start.
# An existing folder is never overwritten: if it already holds DALEEL v2.0.2, that installation is
# started; if it holds another version, or anything else, the command stops and explains what to do.

function Install-DaleelProject {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'  # a progress bar makes Windows PowerShell downloads very slow
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $release = 'v2.0.2'
    $updateGuide = "https://github.com/SarahAlyahya001/DALEEL/blob/$release/docs/ACCOUNT_MANAGEMENT.md#updating-to-a-new-release"
    $target = Join-Path $HOME 'DALEEL'
    if ($env:DALEEL_DIR) { $target = $env:DALEEL_DIR }
    $start = Join-Path $target 'start.bat'

    if (Test-Path (Join-Path $target 'run.py')) {
        # The release of an installation is recorded in its own copy of this script.
        $installer = Join-Path $target 'scripts\install.ps1'
        $installed = ''
        if (Test-Path $installer) {
            $installed = [regex]::Match((Get-Content $installer -Raw), "\`$release = '([^']+)'").Groups[1].Value
        }
        if ($installed -eq $release) {
            Write-Host "DALEEL $release is already installed in $target; starting it."
            & $start
            return
        }
        $found = 'an earlier version of DALEEL'
        if ($installed) { $found = "DALEEL $installed" }
        Write-Host ''
        Write-Host "$target already contains $found." -ForegroundColor Yellow
        Write-Host "This command installs DALEEL $release and never changes an existing installation, its accounts or its missions."
        Write-Host '- To keep using the existing installation, start it from that folder as before.'
        Write-Host "- To move to $release and keep your accounts and missions, follow the update steps at:"
        Write-Host "  $updateGuide"
        Read-Host 'Press Enter to close'
        return
    }
    if ((Test-Path $target) -and (Get-ChildItem $target -Force | Select-Object -First 1)) {
        throw "The folder $target already exists and does not contain DALEEL. Rename or move it, then run the command again."
    }

    Write-Host "Downloading DALEEL $release (30 MB) into $target ..."
    $work = Join-Path ([IO.Path]::GetTempPath()) ("daleel-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force $work | Out-Null
    try {
        $zip = Join-Path $work 'DALEEL.zip'
        try {
            Invoke-WebRequest "https://github.com/SarahAlyahya001/DALEEL/archive/refs/tags/$release.zip" -OutFile $zip -UseBasicParsing
        } catch {
            throw 'DALEEL could not be downloaded from GitHub. Check the internet connection.'
        }
        Expand-Archive $zip (Join-Path $work 'files')
        $source = (Get-ChildItem (Join-Path $work 'files') -Recurse -Filter start.bat | Select-Object -First 1).Directory.FullName
        New-Item -ItemType Directory -Force $target | Out-Null
        Get-ChildItem $source -Force | Move-Item -Destination $target
    } finally {
        Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    }
    Write-Host "DALEEL $release was saved in $target."
    & $start
}

try {
    Install-DaleelProject
} catch {
    Write-Host ''
    Write-Host "DALEEL could not be installed: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host 'Press Enter to close'
}
