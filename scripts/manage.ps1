param([ValidateSet('Setup', 'Repair', 'Start', 'Check')][string]$Action = 'Start')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$desktopEnvironment = Join-Path $projectRoot '.venv'
$desktopPython = Join-Path $desktopEnvironment 'Scripts\python.exe'
try {
    Set-Location -LiteralPath $projectRoot
    if ((Test-Path -LiteralPath $desktopEnvironment) -and
        ((Get-Item -LiteralPath $desktopEnvironment).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'The desktop .venv is a link/junction. Use a local desktop environment; refusing to modify or launch a redirected environment.'
    }
    # Child-process overrides only; never target an activated training environment.
    $env:UV_PROJECT_ENVIRONMENT = $desktopEnvironment
    Remove-Item Env:PYTHONHOME, Env:PYTHONPATH, Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
    if ($Action -in @('Setup', 'Repair')) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw 'uv is not installed. Install it using: winget install --id=astral-sh.uv -e ; then reopen this launcher. See README for alternative installation methods.'
        }
        Write-Host 'Close Mini LoRA Trainer before setup/repair. Only desktop dependencies are changed; training environments and user data are untouched.'
        $syncArguments = @('sync', '--locked', '--inexact', '--python', '3.12')
        if ($Action -eq 'Repair') { $syncArguments += '--reinstall' }
        & uv @syncArguments
        if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed. Check the message above and network access. No training environment was installed.' }
        & $desktopPython -c "from PySide6 import QtWidgets; import PIL, numpy, onnxruntime, huggingface_hub, psutil; print('Desktop imports OK')"
        if ($LASTEXITCODE -ne 0) { throw 'Desktop dependency check failed. Close the app and use Repair.cmd.' }
        Write-Host 'Desktop setup complete. Use Start.cmd. Training needs a separate prepared sd-scripts environment.'
    } else {
        if (-not (Test-Path -LiteralPath $desktopPython -PathType Leaf)) { throw 'Desktop Python is missing. Run Setup.cmd first.' }
        $reportDirectory = Join-Path $projectRoot 'diagnostics'
        New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
        $stamp = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        if ($Action -eq 'Check') {
            $reportPath = Join-Path $reportDirectory ('environment-' + $stamp + '.txt')
            & $desktopPython (Join-Path $projectRoot 'diagnose.py') --report $reportPath
            $checkExitCode = $LASTEXITCODE
            Write-Host "Local report: $reportPath"
            exit $checkExitCode
        }
        & $desktopPython -c 'from PySide6 import QtWidgets; import PIL, numpy, onnxruntime, huggingface_hub, psutil'
        if ($LASTEXITCODE -ne 0) { throw 'Desktop imports failed. Run Setup.cmd (or Repair.cmd), then Check.cmd for a local report.' }
        $stdoutPath = Join-Path $reportDirectory ('startup-' + $stamp + '.log')
        $stderrPath = Join-Path $reportDirectory ('startup-' + $stamp + '.error.log')
        $applicationPath = Join-Path $projectRoot 'app.py'
        $desktopWindowedPython = Join-Path $desktopEnvironment 'Scripts\pythonw.exe'
        if (-not (Test-Path -LiteralPath $desktopWindowedPython -PathType Leaf)) { throw 'Desktop windowed Python is missing. Run Setup.cmd first.' }
        # pythonw avoids a console without SW_HIDE suppressing Qt's first window.
        # This is the interactive application, not a hidden background helper.
        $applicationProcess = Start-Process -FilePath $desktopWindowedPython -ArgumentList ('"' + $applicationPath + '"') -WorkingDirectory $projectRoot -WindowStyle Normal -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
        if ($applicationProcess.WaitForExit(3000) -and $applicationProcess.ExitCode -ne 0) {
            throw "The application exited during startup. Check $stderrPath or run Check.cmd."
        }
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
exit 0
