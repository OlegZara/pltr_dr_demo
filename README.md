# Palantir <> DR integration Repository

This is a clone of the repository created in Foundry. The repository runs the DataRobot python API to train models from Foundry and scoring code/POST request for generating predictions back to a Foundry table. Only files in [this folder](https://github.com/OlegZara/pltr_dr_demo/tree/main/transforms-python/src/myproject/datasets) are created by the user



**Key files to look out for**:

meta.yaml: (dependencies)
https://github.com/OlegZara/pltr_dr_demo/blob/main/transforms-python/conda_recipe/meta.yaml

Example of using DataRobot package for training:
https://github.com/OlegZara/pltr_dr_demo/blob/main/transforms-python/src/myproject/datasets/datarobot_sample.py

Example of using spark based scoring code:
https://github.com/OlegZara/pltr_dr_demo/blob/main/transforms-python/src/myproject/datasets/jar_test.py
