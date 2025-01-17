# Setting Up Your Python Environment

Follow these steps to set up your Python virtual environment and install dependencies:

## 1. Create a Virtual Environment
To create a virtual environment, run the following command:
```bash
python -m venv venv
```

## 2. Activate the Virtual Environment
Activate the virtual environment using the command specific to your operating system:

- **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

## 3. Install Dependencies
Install the required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## 4. Deactivate the Virtual Environment
When you're done, deactivate the virtual environment by running:
```bash
deactivate
```

## Notes
- Ensure you have Python 3.7 or higher installed on your system.
- To check your Python version, use:
  ```bash
  python --version
  ```
