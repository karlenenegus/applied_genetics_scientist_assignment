# Pipeline Description

To productionize this project as a cloud pipeline, I would:

1. Package this project into a Docker image to increase reproducibility.
2. Add functionality for pulling from additional data sources like SQL databases and improve generalized handling of data to work with diverse data sources.
3. Implement continuous integration/continuous deployment checks to appropriately validate code updates before deployment. 
4. Update configuration handling to allow for overrides from the command line.
5. Add additional data checks for running predictions using existing models on new datasets and streamline the user interface for running predictions on pretrained models.
6. Improve the modularity of the script steps so the user interface is easier to use them more independently.