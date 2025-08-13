# YOLO Divot Detection Project

This project uses a YOLO model to detect divots in videos.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/teomeo12/yolo-divot-detection.git
    cd yolo-divot-detection
    ```

2.  **Install Dependencies:**
    Make sure you have Python and the required libraries installed.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created to list dependencies like PyTorch, OpenCV, etc.)*

3.  **Add Model Files:**
    This project requires pre-trained model files (`.pt`) that are not tracked by Git. You will need to manually add them to the correct project folders.

    For example, place your model file in the corresponding directory:
    ```
    /yolo11s_1600_100ep/your_model.pt
    ```

4.  **Run the Script:**
    To process videos, run the `divot_from_video.py` script:
    ```bash
    python yolo11s_1600_100ep/divot_from_video.py
    ```

## Usage

-   Place the videos you want to process in the `/videos` directory.
-   Update the `project_models` dictionary in `divot_from_video.py` to select which models to use.
-   Processed videos will be saved in the `processed_videos` subfolder within each model's project directory. 