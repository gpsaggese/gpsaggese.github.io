# Ansible

## Description
- Ansible is an open-source automation tool used for application deployment,
  configuration management, and task automation.
- It uses a simple, human-readable YAML syntax to define automation tasks,
  making it accessible for users without extensive programming knowledge.
- Ansible operates in an agentless manner, meaning it does not require any
  software to be installed on the target machines, allowing for easier
  management of systems.
- It supports a wide range of modules for various tasks, including cloud
  provisioning, orchestration, and security compliance, enabling extensive
  automation capabilities.
- Ansible is designed to be idempotent, which means that running the same
  playbook multiple times will not change the system beyond the initial
  application, ensuring stability and predictability.

## Project Objective
The goal of the project is to automate the deployment of a machine learning
model using Ansible. Students will create a playbook that provisions a virtual
machine, installs necessary dependencies, and deploys a pre-trained model to
serve predictions via a REST API. The project will optimize the deployment
process to ensure it is efficient and reproducible.

## Dataset Suggestions
1. **Kaggle House Prices Dataset**
   - **Source Name**: Kaggle
   - **URL**:
     [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Various features of houses in Ames, Iowa, including sale
     prices, which can be used for regression tasks.
   - **Access Requirements**: Free account on Kaggle.

2. **UCI Machine Learning Repository: Wine Quality Dataset**
   - **Source Name**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties of wine samples along with quality
     ratings, suitable for classification tasks.
   - **Access Requirements**: No authentication required.

3. **Open Government Data: NYC Taxi Trip Data**
   - **Source Name**: NYC Open Data
   - **URL**: [NYC Taxi Trip Data](https://opendata.cityofnewyork.us/)
   - **Data Contains**: Trip records including pickup and drop-off locations,
     times, and fares, which can be used for regression or clustering tasks.
   - **Access Requirements**: Publicly available without authentication.

## Tasks
- **Set Up Virtual Environment**: Create a virtual machine using Ansible to host
  the machine learning model.
- **Install Dependencies**: Write Ansible tasks to install necessary libraries
  and frameworks (e.g., Flask, scikit-learn) for serving the model.
- **Deploy Model**: Use Ansible to copy the pre-trained model files to the
  virtual machine and configure the application to serve predictions.
- **Create REST API**: Implement a simple REST API using Flask to handle
  incoming prediction requests and return results.
- **Testing and Validation**: Write Ansible tasks to test the deployment and
  validate that the API is returning the expected outputs.

## Bonus Ideas
- **Monitoring and Logging**: Extend the project by integrating monitoring tools
  (e.g., Prometheus) to keep track of API performance and logs.
- **Scaling Deployment**: Explore how to scale the deployment across multiple
  servers using Ansible's orchestration capabilities.
- **CI/CD Pipeline**: Implement a continuous integration/continuous deployment
  (CI/CD) pipeline to automate updates to the model and application.

## Useful Resources
- [Ansible Documentation](https://docs.ansible.com/ansible/latest/index.html)
- [Ansible GitHub Repository](https://github.com/ansible/ansible)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Flask Documentation](https://flask.palletsprojects.com/)
