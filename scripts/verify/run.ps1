<#
  Production Pulse - staged verification on Windows.
  Every stage records its exact command, start time, duration and exit code in
  logs\verification\summary.txt, and its stdout/stderr in logs\verification\<stage>.out/.err.txt.
  Stages are run ONE AT A TIME (no parallel memory-heavy work).

  Usage (from the project root, current PowerShell process only - no system setting is changed):
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify\run.ps1 preflight
  Stages: preflight | pytest | nb01 | nb02 | nb03 | nb04 | manifest | api | frontend
  Exit code: 0 if every command of the stage exited 0, otherwise the first non-zero exit code
  (also written as a RESULT line in summary.txt).
  api: starts its OWN server on a free port (never touches a server you already run, e.g. on 8000).
    Pick a port with  $env:PP_API_SMOKE_PORT = "8011"  (default "auto"); exit 3 = port not available.

  Executed notebooks are historical evidence and are NEVER overwritten. nb02 writes
  02_synthetic_anomaly_validation.corrected.executed.ipynb, nb04 writes 04_equal_fpr_comparison.executed.ipynb;
  if the target file already exists, a time-stamped name is used instead. nb03 retrains the model, so it is
  skipped while artifacts\models\autoencoder_v1\model.keras exists (move or rename that folder first if you
  really want to retrain; document why).
#>
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("preflight", "pytest", "nb01", "nb02", "nb03", "nb04", "manifest", "api", "frontend")]
  [string]$Stage
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = Join-Path $Root ".venv313\Scripts\python.exe"
$LogDir = Join-Path $Root "logs\verification"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Summary = Join-Path $LogDir "summary.txt"

function Invoke-Stage {
  param([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$WorkDir = $Root, [hashtable]$EnvVars = @{})
  $out = Join-Path $LogDir "$Name.out.txt"
  $err = Join-Path $LogDir "$Name.err.txt"
  $saved = @{}
  foreach ($k in $EnvVars.Keys) {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
    [Environment]::SetEnvironmentVariable($k, $EnvVars[$k], "Process")
  }
  $start = Get-Date
  $code = $null
  try {
    $p = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkDir -NoNewWindow -PassThru `
      -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle          # cache the handle so ExitCode is available after the wait
    $p.WaitForExit()
    $code = $p.ExitCode
  } catch {
    $code = -1
    ($_ | Out-String) | Set-Content -Path $err -Encoding UTF8
  } finally {
    foreach ($k in $EnvVars.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], "Process") }
  }
  $secs = ((Get-Date) - $start).TotalSeconds
  $envText = if ($EnvVars.Count) { " env=" + (($EnvVars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ",") } else { "" }
  $line = "{0}  stage={1}  exit={2}  seconds={3:N1}  cwd={4}{5}  cmd=`"{6}`" {7}" -f $start.ToString("s"), $Name, $code, $secs, $WorkDir, $envText, $FilePath, ($Arguments -join " ")
  Add-Content -Path $Summary -Value $line -Encoding UTF8
  Write-Host $line
  if ($code -ne 0 -and $script:StageExit -eq 0) { $script:StageExit = $code }
  return $code
}

function Invoke-Notebook([string]$Name, [string]$Notebook, [string]$OutputBase = "") {
  $base = [IO.Path]::GetFileNameWithoutExtension($Notebook)
  if ($OutputBase -eq "") { $OutputBase = "$base.executed" }
  $target = Join-Path $Root "reports\executed\$OutputBase.ipynb"
  if (Test-Path $target) {
    # never overwrite an executed notebook (historical evidence)
    $OutputBase = "$OutputBase.rerun-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Write-Host "Executed copy already exists; writing to $OutputBase.ipynb instead."
  }
  # JUPYTER_PATH puts .venv313's python3 kernelspec first (child process only)
  Invoke-Stage $Name $Py @("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "notebooks\$Notebook",
    "--output-dir", "reports\executed", "--output", "$OutputBase.ipynb",
    "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3") `
    -EnvVars @{ JUPYTER_PATH = (Join-Path $Root ".venv313\share\jupyter"); PYTHONDONTWRITEBYTECODE = "1" } | Out-Null
}

Set-Location $Root
$script:StageExit = 0
Add-Content -Path $Summary -Value "---- $(Get-Date -Format s) begin stage '$Stage' ----" -Encoding UTF8

switch ($Stage) {
  "preflight" {
    $sys = Join-Path $LogDir "00_system.txt"
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $drive = (Get-Item $Root).PSDrive
    @(
      "Collected: $(Get-Date -Format s)",
      "OS: $($os.Caption) $($os.Version) ($($os.OSArchitecture))",
      ("RAM total GB: {0:N2}" -f ($cs.TotalPhysicalMemory / 1GB)),
      ("RAM free GB: {0:N2}" -f ($os.FreePhysicalMemory * 1KB / 1GB)),
      "Logical CPUs: $($cs.NumberOfLogicalProcessors)",
      ("Disk {0}: free GB {1:N1} of {2:N1}" -f $drive.Name, ($drive.Free / 1GB), (($drive.Used + $drive.Free) / 1GB)),
      "PowerShell: $($PSVersionTable.PSVersion)",
      "Python exe exists: $(Test-Path $Py) ($Py)"
    ) | Set-Content -Path $sys -Encoding UTF8
    Get-Content $sys | Write-Host
    Invoke-Stage "00_python_version" $Py @("--version") | Out-Null
    Invoke-Stage "00_check_env" $Py @("scripts\verify\check_env.py") | Out-Null
    Invoke-Stage "00_checksums" $Py @("scripts\verify_checksums.py") | Out-Null
    Invoke-Stage "00_kernelspecs" $Py @("-m", "jupyter", "kernelspec", "list") -EnvVars @{ JUPYTER_PATH = (Join-Path $Root ".venv313\share\jupyter") } | Out-Null
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) { Invoke-Stage "00_node_version" $node.Source @("--version") | Out-Null } else { Add-Content $Summary "stage=00_node_version  RESULT=node NOT FOUND on PATH" }
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npm) { Invoke-Stage "00_npm_version" $npm.Source @("--version") | Out-Null } else { Add-Content $Summary "stage=00_npm_version  RESULT=npm NOT FOUND on PATH" }
  }
  "pytest" {
    Invoke-Stage "10_pytest_full" $Py @("-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfEs", "--junitxml=logs\verification\10_pytest_full.xml") -EnvVars @{ PYTHONDONTWRITEBYTECODE = "1" } | Out-Null
    Invoke-Stage "11_pytest_realdata" $Py @("-m", "pytest", "-p", "no:cacheprovider", "-q", "-rA", "--junitxml=logs\verification\11_pytest_realdata.xml", "tests\test_real_data.py", "tests\test_api_offline_parity.py", "tests\test_demo_prediction.py") -EnvVars @{ PP_RUN_REALDATA = "1"; PYTHONDONTWRITEBYTECODE = "1" } | Out-Null
  }
  "nb01" { Invoke-Notebook "20_nb01" "01_data_exploration.ipynb" }
  "nb02" { Invoke-Notebook "21_nb02" "02_synthetic_anomaly_validation.ipynb" "02_synthetic_anomaly_validation.corrected.executed" }
  "nb03" {
    if (Test-Path (Join-Path $Root "artifacts\models\autoencoder_v1\model.keras")) {
      $msg = "stage=22_nb03  RESULT=SKIPPED: a trained model already exists; notebook 03 would retrain it and cannot save over it."
      Add-Content -Path $Summary -Value $msg -Encoding UTF8
      Write-Host $msg
    } else { Invoke-Notebook "22_nb03" "03_autoencoder_training.ipynb" }
  }
  "nb04" { Invoke-Notebook "23_nb04" "04_equal_fpr_comparison.ipynb" "04_equal_fpr_comparison.executed" }
  "manifest" { Invoke-Stage "24_manifest" $Py @("scripts\write_artifact_manifest.py") -EnvVars @{ PYTHONDONTWRITEBYTECODE = "1" } | Out-Null }
  "api" {
    $smokeEnv = @{ PYTHONDONTWRITEBYTECODE = "1" }
    if ($env:PP_API_SMOKE_PORT) { $smokeEnv["PP_API_SMOKE_PORT"] = $env:PP_API_SMOKE_PORT }   # logged in summary.txt
    Invoke-Stage "30_api_smoke" $Py @("scripts\verify\api_smoke.py") -EnvVars $smokeEnv | Out-Null
  }
  "frontend" {
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $fe = Join-Path $Root "frontend"
    $c = Invoke-Stage "40_npm_install" $npm @("install", "--no-audit", "--no-fund") -WorkDir $fe
    if ($c -eq 0) {
      Invoke-Stage "41_npm_typecheck" $npm @("run", "typecheck") -WorkDir $fe | Out-Null
      Invoke-Stage "42_npm_test" $npm @("test") -WorkDir $fe | Out-Null
      Invoke-Stage "43_npm_build" $npm @("run", "build") -WorkDir $fe | Out-Null
    }
  }
}
Add-Content -Path $Summary -Value "---- $(Get-Date -Format s) end stage '$Stage'  RESULT exit=$($script:StageExit) ----" -Encoding UTF8
Write-Host "Done (exit $($script:StageExit)). Logs in $LogDir"
exit $script:StageExit
