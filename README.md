This guide will help you set up your Python environment and configure Webots to run your project on both **Windows and Linux**.

## 1. Environment Setup

Navigate to the root directory of your Git repository in your terminal and execute the following commands. These commands work for both Windows and Linux.

```bash
# Install the uv package manager
pip install uv
# Create a virtual environment in a .venv directory
uv venv
```

After setting up the environment, you need to activate it.

- **On Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```
- **On Windows (Command Prompt/PowerShell):**
  ```bash
  .venv\Scripts\activate
  ```

Then sync dependencies
```bash
  uv sync
```

## 2. Webots Configuration
1. **Open `worlds/world.wbt` in Webots**

2.  **Locate your Virtual Environment's Python Executable:**

    First, make sure your virtual environment is activated by following the instructions at the end of the previous section. Then, run the appropriate command for your operating system to find the absolute path to its Python executable.

    - **On Linux/macOS:**

      ```bash
      which python
      ```

      Copy this path. It will look something like `/path/to/your/repo/.venv/bin/python`.

    - **On Windows:**
      ```bash
      where python
      ```
      Copy the path that points to the `.venv` directory. It will look something like `C:\path\to\your\repo\.venv\Scripts\python.exe`.

3.  **Open Webots Preferences:**
    In Webots, go to `Tools > Preferences`.
    ![alt text](image.png)

4.  **Paste the Python Path:**
    In the Preferences window, find the `Python Command:` field and paste the absolute path you copied in the previous step.
    ![alt text](image-2.png)

## 3. Running the Controller

Once your environment is set up and Webots is configured, you can run your robot's controller within Webots.
![alt text](image-3.png)

