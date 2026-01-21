# Virtual Environment Activation Guide / 虚拟环境激活指南

This project uses `uv` to manage the virtual environment. The environment is located in the `.venv` directory.
本项目使用 `uv` 管理虚拟环境。环境位于 `.venv` 目录中。

## Activation / 激活方法

Choose the command corresponding to your operating system and shell.
请选择对应您操作系统和终端的命令。

### Windows

**PowerShell:**
```powershell
.\.venv\Scripts\activate
```
*Note: If you encounter execution policy errors, you may need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.*
*注意：如果遇到执行策略错误，可能需要先运行 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`。*

**Command Prompt (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**Git Bash (Windows):**
```bash
source .venv/Scripts/activate
```

### macOS / Linux

**Bash / Zsh:**
```bash
source .venv/bin/activate
```
