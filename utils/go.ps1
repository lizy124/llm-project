# go.ps1 - remote command runner (double base64, quoting-proof)
# Usage: pass the whole remote command as ONE PowerShell string (prefer single quotes).
#   .\go.ps1 'docker exec cxy_cann9.1.0 bash -c "pip show vllm | grep Version"'
#   .\go.ps1 -Server root@192.168.13.165 'npu-smi info | grep -c 0000:'
# See remote_exec_guide.md for the rationale.
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Command,
    [string]$Server = "root@192.168.13.165"
)

function ConvertTo-B64([string]$s) {
    return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($s))
}

# Layer 1: encode the user command as-is.
# Layer 2: encode the wrapper script that decodes layer 1 and pipes it to bash.
# Only base64 characters travel through PowerShell -> ssh -> remote bash.
$layer1 = ConvertTo-B64 $Command
$layer2 = ConvertTo-B64 "echo $layer1 | base64 -d | bash"

ssh -o BatchMode=yes $Server "echo $layer2 | base64 -d | bash"
exit $LASTEXITCODE
