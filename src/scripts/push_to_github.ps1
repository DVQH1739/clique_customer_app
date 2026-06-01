# Create GitHub repo and push (run after: gh auth login)
$ErrorActionPreference = "Stop"
$GH = "C:\Program Files\GitHub CLI\gh.exe"
$GIT = "C:\Program Files\Git\cmd\git.exe"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

& $GH auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Run: gh auth login -h github.com -p https -w"
    exit 1
}

$RepoName = "clique-customer-app"
& $GH repo create $RepoName --public --source=. --remote=origin --push --description "CLIQUE subspace clustering for customer profiles (Streamlit + Online Retail II)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "If repo exists, try: git remote add origin https://github.com/YOUR_USER/$RepoName.git"
    & $GIT remote add origin "https://github.com/$(& $GH api user -q .login)/$RepoName.git" 2>$null
    & $GIT push -u origin main
}

Write-Host "Done. View at: https://github.com/$(& $GH api user -q .login)/$RepoName"
