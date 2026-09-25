# Record Python and package environment for the publication reproducibility audit.
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\19_record_publication_environment.ps1

$ErrorActionPreference = "Stop"

$outdir = "outputs\tables"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

python --version 2>&1 |
    Set-Content "$outdir\publication_python_version.txt" -Encoding utf8

pip freeze |
    Set-Content "$outdir\publication_environment.txt" -Encoding utf8

Write-Host ""
Write-Host "Environment recorded:"
Write-Host "  $outdir\publication_python_version.txt"
Write-Host "  $outdir\publication_environment.txt"
Write-Host ""
Get-Content "$outdir\publication_python_version.txt"
