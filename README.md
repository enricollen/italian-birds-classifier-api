# italian-birds-classifier-api
A simple python Flask API which predicts the bird species (Note: italian species only) given an image. 

The model is able to recognize 387 species with an avg accuracy of 80.05%. It was built upon training an EfficientNet on 69k scraped images (copyright-free), exploiting transfer learning + fine tuning.

Please try to upload images of birds which are as meaningful as possibile, without too much uncertainty (portaits or side pictures are preferred), so that will be easier for the model to infer the right species.

The model particularly suffers from missclassification when it has to predict between birds which are very similar, e.g., same species and different family, or between male/female across different species.
