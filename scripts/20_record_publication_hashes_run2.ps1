# Record SHA-256 hashes after the publication reproducibility rerun.
# Run from the repository root AFTER rerunning scripts 15, 16 and 17:
#   powershell -ExecutionPolicy Bypass -File scripts\20_record_publication_hashes_run2.ps1

$ErrorActionPreference = "Stop"

$files = @(
    "outputs\tables\_pit_real.csv",
    "src\tailrisk\inference.py",
    "scripts\15_publication_stress_combined.py",
    "scripts\16_publication_stress_disaggregated.py",
    "scripts\17_diagnose_brent_wti_covid.py",
    "outputs\tables\calm_stress_full60_B15000.csv",
    "outputs\tables\calm_stress_disaggregated_B15000.csv",
    "outputs\tables\diagnostic_brent_wti_covid_upper.csv"
)

$missing = $files | Where-Object { -not (Test-Path $_) }

if ($missing) {
    Write-Host "ERROR: The following required files are missing:"
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}

$outfile = "outputs\tables\publication_hashes_run2.txt"

Get-FileHash $files -Algorithm SHA256 |
    ForEach-Object {
        "$($_.Hash.ToLower())  $($_.Path)"
    } |
    Set-Content $outfile -Encoding utf8

Write-Host ""
Write-Host "Run 2 hashes written to:"
Write-Host "  $outfile"
Write-Host ""
Write-Host "Recorded hashes:"
Get-Content $outfile
