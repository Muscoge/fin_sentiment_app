# Financial Sentiment Prediction App
#### By Egor Sergeev, HSE MDS Student Spring23
This application is designed to predict financial message sentiment using fine-tuned machine learning models. Below are the details of the project's implementation and usage.

## Methodology

1. **Data Preprocessing**:
   - The text data is tokenized without additional preprocessing, except for converting text to lowercase. 
   - This approach preserves critical symbols that may influence sentiment detection in financial contexts.

2. **Model**:
   - The application uses a **BERT model fine-tuned on financial data** for sentiment prediction.
   - The fine-tuning process is implemented in the `finetune.py` script.

3. **Experiment Tracking**:
   - All experiments are tracked using **MLFlow**. 
   - Logs and artifacts are stored in the `mlruns` folder.

## Application Structure

Application implementation is stored in `app.py` file. 

1. **Backend**:
   - A **Flask application** powers the backend, offering both an API and a direct host IP interface.

2. **API Usage**:
   - Users can interact with the model via API by sending requests to the server.

### Example API Request (Local Machine)

```bash
curl -X POST -F "file=@C:\HSE_MDS\LSML_2\Final_Project\input.txt" http://localhost:8000/api/predict
```
### Direct Host Usage

The application can also be accessed by opening the host IP directly in a browser.

## Deployment

### Docker Image

For ease of use and reproducibility, a pre-built Docker image is available on Docker Hub:

```bash
docker run muscorgi/fin-sentiment-app:latest
```
### Building the Docker Image Locally

If needed, users can build the image locally using the provided `Dockerfile`:

```bash
docker build -t fin-sentiment-app .
```
## Summary

The Financial Sentiment Prediction App offers a streamlined solution for analyzing financial sentiment using state-of-the-art machine learning techniques. Its robust architecture, tracked experiments, and Docker-based deployment ensure ease of use and reproducibility.


