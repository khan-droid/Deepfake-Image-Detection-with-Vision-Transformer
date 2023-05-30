# Deepfake-Image-Detection-with-Vision-Transformer
A comprehensive deep learning-based solution designed to detect deepfake images using the power of Vision Transformer

With the increasing prevalence of synthetic media and the potential risks it poses to privacy, democracy, and national security, DeepVision aims to provide an efficient and accurate means of identifying manipulated images.

The system divides the input image into fixed-size patches and generates a sequence of vectors, which are then linearly embedded and enriched with position embeddings. Finally, the assembled vectors are fed into a conventional Transformer encoder to classify the image as genuine or deepfake.

The DeepVision project goes beyond the algorithm itself by integrating it into a user-friendly graphical user interface (GUI) developed using Django. This GUI provides a seamless experience for users to conveniently input their images, initiate the deepfake detection process, and visualize the results. The intuitive interface ensures accessibility for both technical and non-technical users, enabling a wide range of individuals to utilize this powerful deepfake detection tool.

### Dataset
I used 140k Real and Fake faces dataset. This dataset includes all 70k REAL faces from the Flickr dataset gathered by Nvidia as well as a sample of 70k FAKE faces drawn from Bojan's 1 Million FAKE faces (created using StyleGAN)

### Key Features:

- Integration of Vision Transformer for accurate deepfake image detection
- User-friendly GUI built with Django for easy image input and result visualization
- Efficient image processing and analysis pipeline for quick detection
- Real-time feedback and informative visualizations for enhanced user experience
- Robust architecture capable of handling a large number of images with high performance
- Potential for scalability and adaptability to future advancements in deepfake detection techniques
- DeepVision provides a crucial defense against the rapid proliferation of deepfake images, helping to preserve privacy, safeguard democracy, and maintain national security. By harnessing the capabilities of Vision Transformer and providing a user-friendly interface, this project empowers users to identify and combat the threats posed by synthetic media.

## Steps
1. Run the google colab notebook to train and download the model in '.h5' format
2. Paste the model file in '.models/'
3. Run the django application
```sh
python manage.py runserver
```
