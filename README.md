# Training DCGAN to generate Pokemons and deploying locally using Streamlit.

## Dataset
Pokemon dataset used: https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset

## File/Directories

- training_GAN.ipynb: Contains
    - code for training GAN model on Pokemon dataset for different epochs. (50, 100, 250, 500)
    - Plots of generated images at different epochs also present here.

- logs:
    - logs for different training epochs are saved here.

- output: contains generated images after different training epochs.

- models: contains models after different training epochs.

- pokemon/pokemon_jpg: contains pokemon images (Dataset)

- frontend.py: streamlit app to generate images using trained model.

