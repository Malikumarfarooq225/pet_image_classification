# Advanced Image Classification using Oxford-IIIT Pet Dataset

This project uses a MobileNetV2 model to classify pet breeds using the Oxford-IIIT Pet Dataset. The dataset contains images of various pet breeds, and the model is trained to predict the breed of a given pet image.

## Folder Structure

- **/data**: Contains the Oxford-IIIT Pet Dataset.
- **/images**: Folder for storing processed images (optional).
- **/models**: Saves the model checkpoints (`model.pth`).
- **/scripts**: Contains all the Python scripts for training, evaluation, and preprocessing.
- **/notebooks**: (Optional) Jupyter notebooks for dataset exploration and visualization.

## Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Advanced_Image_Classification_Pet_Dataset.git
    cd Advanced_Image_Classification_Pet_Dataset
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

      ```bash
      .\venv\Scripts\activate
      ```

    - On macOS/Linux:

      ```bash
      source venv/bin/activate
      ```

4. **Install required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Download the Oxford-IIIT Pet Dataset** from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/) and place it in the `/data` folder.

6. **Training the Model:**

    To train the model, run the following command:

    ```bash
    python scripts/main.py --train
    ```

7. **Evaluating the Model:**

    After training, you can evaluate the model's performance on the validation set:

    ```bash
    python scripts/main.py --evaluate
    ```

8. **Running Both Training and Evaluation:**

    You can also run both training and evaluation in one command:

    ```bash
    python scripts/main.py --train --evaluate
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
