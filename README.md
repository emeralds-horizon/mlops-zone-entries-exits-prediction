# WP4 MAIaaS Platform: T4.3 - MLOps Kserve Template

Template model Deployment

## Authors

ATOS-Eviden

## Things to Keep in Mind

There are two environments: development (staging branch) and production (main branch). The main idea is that staging can receive changes from any branch, but the main branch only receives changes from staging. Both branches, when a `git push` is executed, will deploy an inference model to the Kubernetes cluster (this process typically takes around 10 minutes; you can monitor it in the "Actions" tab).

## How to Use This Repository

Before running an inference service on the MAIaaS cluster, please contact our MLOps Engineer at [andres.cardoso@eviden.com](mailto:andres.cardoso@eviden.com).

Once you’re ready to make updates (e.g., to `model.py`, `Dockerfile`, `pyproject.toml`, or any files in `test_request/*.py`):

1. **Create a feature branch** for your changes.
2. **Open a pull request (PR) to the `staging` branch** and specify that you want to deploy in staging. The MLOps Engineer will then trigger the deployment workflow by merging your PR.
3. **After successful testing**, the MLOps Engineer will create another PR from `staging` to `main` to finalize and deploy the inference service.

### Important Files for Creating Inference Models

- **model.py**: Contains the logic for prediction, preprocessing, and loading of pre-trained models (whether from Jupyter notebooks, Kubeflow, or other sources).
- **test_request**: This directory includes three files:
  - **kserve_request_production.py**: Used to make requests to the production inference model.
  - **kserve_request_staging.py**: Used to make requests to the development inference model.
  - **local_request.py**: Used for local testing. Run `model.py` in another terminal before using this.
- **pyproject.toml**: Lists the Python libraries required for the project.
- **poetry.lock**: Ensures the best combinations of library versions are used.
- **Other configuration files**: These are managed by a Kubernetes engineer and include `.workflows`, the `kubernetes` folder, and the `Dockerfile`.

### Endpoints

- **Main/Production Environment**:
  - `KSERVE_MODEL_ENDPOINT`: "https://model-example.models.kubeflow.emeralds.ari-aidata.eu/v1/models/model-example:predict"
- **Staging/Development Environment**:
  - `KSERVE_MODEL_ENDPOINT`: "https://model-example.stagingv2.kubeflow.emeralds.ari-aidata.eu/v1/models/model-example:predict"
