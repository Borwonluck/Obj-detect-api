# Bitterbean Detection API

A FastAPI web service for detecting bitterbean (Parkia speciosa) seeds and their ripeness using a YOLO deep learning model.

## Features

- Detect bitterbean seeds and classify ripeness (ripe, unripe, overripe)
- Measure object size in centimeters
- Store results in a PostgreSQL database
- Simple REST API for detection and history

## Installation

1. **Clone the repository**
    ```
    git clone https://github.com/yourusername/bitterbean_api.git
    cd bitterbean_api
    ```

2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Configure the database**

    Edit `bitterbean_api.py` if you need to change database credentials:
    ```python
    DB_USER = "postgres"
    DB_PASS = "your_password"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "bitterdb"
    ```

4. **Add your YOLO model**

    Place `my_model.pt` in the same directory as the API code.

5. **Run the API**
    ```
    uvicorn bitterbean_api:app --host 0.0.0.0 --port 9000 --reload
    ```

## Usage

- **POST `/detect`**  
  Upload an image for detection. Example using curl:
    ```
    curl -X POST "http://localhost:9000/detect" -F "file=@your_image.jpg"
    ```

- **GET `/history`**  
  Retrieve detection history.

## Requirements

- Python 3.9+
- FastAPI
- SQLAlchemy
- PostgreSQL
- PyTorch
- OpenCV
- ultralytics (YOLO)
- Pillow

## License

MIT License

