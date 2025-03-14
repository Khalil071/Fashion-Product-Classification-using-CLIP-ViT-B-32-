Fashion Product Classification using CLIP (ViT-B/32)

Overview

This project implements a CLIP (Contrastive Language-Image Pretraining) model for detecting and classifying fashion products from the ceyda/fashion-products-small dataset. The model used is:

CLIPModel: CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

CLIPProcessor: Used for preprocessing images and text before feeding them into the model.

Fine-tuned CLIP Model: A customized version of CLIP fine-tuned on the fashion dataset to improve classification accuracy.

CLIP enables zero-shot classification by matching images with textual descriptions, making it highly effective for fashion product categorization.

Dataset

The ceyda/fashion-products-small dataset contains labeled images of various fashion products. The dataset is structured as follows:

├── dataset/
│   ├── images/
│   │   ├── Tops/
│   │   ├── Bottoms/
│   │   ├── Shoes/
│   │   ├── Accessories/
│   │   ├── ...

Dataset Source: Hugging Face - ceyda/fashion-products-small

Project Structure

├── dataset/                     # Contains fashion product images
├── models/                      # Saved trained models (including fine-tuned CLIP model)
├── notebooks/                   # Jupyter notebooks for experimentation
├── src/                         # Source code
│   ├── clip_model.py            # CLIP model implementation
│   ├── train.py                 # Fine-tuning script
│   ├── evaluate.py              # Model evaluation script
│   ├── preprocess.py            # Preprocessing functions for images and text
├── requirements.txt             # Required dependencies
├── README.md                    # Project documentation

Installation

Clone the repository:

git clone https://github.com/your-username/clip-fashion-classification.git
cd clip-fashion-classification

Create a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Model Usage

Run inference using the pre-trained CLIP model:

python src/clip_model.py --image_path path/to/image.jpg --labels "Tops, Bottoms, Shoes, Accessories"

Run inference using the fine-tuned CLIP model:

python src/clip_model.py --image_path path/to/image.jpg --model_path models/fine_tuned_clip --labels "Tops, Bottoms, Shoes, Accessories"

Fine-Tuning

Fine-tune CLIP on the fashion dataset for improved accuracy:

python src/train.py --dataset dataset/ --output_model_path models/fine_tuned_clip

Evaluation

Evaluate the model's performance on a test set:

python src/evaluate.py --dataset dataset/

Results

Results include classification accuracy, similarity scores between images and text labels, and visualizations. Detailed analysis is available in the Jupyter notebooks under notebooks/.

Future Improvements

Experimenting with other CLIP models like ViT-L/14 for better accuracy.

Further fine-tuning CLIP on more extensive fashion datasets.

Deploying a fashion recommendation system based on CLIP embeddings.

Contributing

Feel free to contribute by opening issues or submitting pull requests!

License

This project is licensed under the MIT License.
