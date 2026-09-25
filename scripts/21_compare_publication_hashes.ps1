# Compare publication Run 1 and Run 2 SHA-256 hash manifests.
# Run from the repository root AFTER scripts\20_record_publication_hashes_run2.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\21_compare_publication_hashes.ps1

$ErrorActionPreference = "Stop"

$run1 = "outputs\tables\publication_hashes_run1.txt"
$run2 = "outputs\tables\publication_hashes_run2.txt"

if (-not (Test-Path $run1)) {
    Write-Host "ERROR: Missing $run1"
    exit 1
}
if (-not (Test-Path $run2)) {
    Write-Host "ERROR: Missing $run2"
    exit 1
}

$diff = Compare-Object (Get-Content $run1) (Get-Content $run2)

if ($null -eq $diff) {
    Write-Host ""
    Write-Host "PASS: Run 1 and Run 2 hashes match exactly."
    Write-Host "All tracked publication inputs, code files and outputs are byte-for-byte reproducible."
    exit 0
}

Write-Host ""
Write-Host "FAIL: Run 1 and Run 2 differ:"
$diff | Format-Table -AutoSize
exit 1
