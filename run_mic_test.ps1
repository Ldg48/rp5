Set-Location -Path $PSScriptRoot
python tools\transcribe_mic_int8.py --seconds 5 --device 1
Read-Host "Press Enter to close"
