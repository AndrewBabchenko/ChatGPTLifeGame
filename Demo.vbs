' Launch Simulation Demo without console window
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Try rocm venv first, then regular venv
If CreateObject("Scripting.FileSystemObject").FileExists(".venv_rocm\Scripts\pythonw.exe") Then
    WshShell.Run """.venv_rocm\Scripts\pythonw.exe"" scripts\run_demo.py", 0, False
ElseIf CreateObject("Scripting.FileSystemObject").FileExists(".venv\Scripts\pythonw.exe") Then
    WshShell.Run """.venv\Scripts\pythonw.exe"" scripts\run_demo.py", 0, False
Else
    WshShell.Run "pythonw scripts\run_demo.py", 0, False
End If
